"""gpu-mode replica: one engine process per card, each running the whole model, behind the one engine a Service drives.

A request goes to the replica whose recent prompts share the longest start with it, so a conversation keeps reading only
its new tokens on the card that holds the rest. A turn that continues a long earlier prompt waits for the card that read
it rather than read it all again on another; any other prompt goes to a replica with a free slot, the one holding most of
its start first, else the least busy. Every lifecycle call (start, sleep, wake, VRAM, unload) applies to every replica;
the slots are numbered card after card.
"""
from __future__ import annotations

import collections
import contextlib
import threading
import time
from pathlib import Path

from serve.engine.flags import flag_index, flag_value

BLOCK = 256              # prompt ids per routing block (characters for a text prompt: four times as many)
BLOCKS_KEPT = 1 << 18    # block hashes remembered per replica, least recently used dropped first
ENDS_KEPT = 4096         # whole prompts remembered per replica
SESSIONS_KEPT = 4096     # client sessions remembered
STAY_BLOCKS = 16         # a turn continuing an earlier prompt this long waits for that prompt's card
ENCODE_POLL_S = 0.05     # how often an encode looks for a card free of requests
WAITING = object()       # a request's `seen` until it holds its replica: no DONE is its own yet


class Prompt:
    """What routing compares of a prompt (its ids, or the rendered text the engine tokenizes): one hash per whole block,
    each covering the blocks before it too, so equal hashes mean an equal start; and the whole prompt's own mark."""

    def __init__(self, ids):
        text = getattr(ids, "text", None)
        self.seq, size = (text, BLOCK * 4) if isinstance(text, str) else (ids, BLOCK)
        self.size = size
        self.hashes, h = [], 0
        for i in range(0, len(self.seq) - size + 1, size):
            h = self.hash(h, self.seq[i:i + size])
            self.hashes.append(h)
        tail = self.seq[len(self.hashes) * size:]
        self.end = (self.hashes[-1], len(tail), self.hash(0, tail)) if self.hashes else None

    @staticmethod
    def hash(h: int, part) -> int:
        return hash((h, part if isinstance(part, str) else tuple(part)))

    def continues(self, end) -> int:
        """The blocks of an earlier prompt (its `end` mark) this prompt starts with in full; 0 when it does not."""
        last, n, tail = end
        at = self.index.get(last)
        if at is None:
            return 0
        start = (at + 1) * self.size
        return at + 1 if len(self.seq) >= start + n and self.hash(0, self.seq[start:start + n]) == tail else 0

    @property
    def index(self) -> dict:
        if not hasattr(self, "_index"):
            self._index = {h: i for i, h in enumerate(self.hashes)}
        return self._index


SHARED_BUDGETS = ("--conversation-cache-mib", "--conversation-cache-disk-mib")   # the machine's RAM and disk, split


def replica_args(args: list[str], name: str, count: int) -> list[str]:
    """A replica's engine arguments: its own folder for the conversations it parks on disk, and its part of the RAM
    and disk budgets the replicas share."""
    out = list(args)
    where = flag_value(out, "--conversation-cache-dir")
    if where:
        out[flag_index(out, "--conversation-cache-dir") + 1] = str(Path(where) / name)
    for flag in SHARED_BUDGETS:
        value = flag_value(out, flag)
        if value and value.isdigit() and int(value) > 0:
            out[flag_index(out, flag) + 1] = str(max(1, int(value) // count))
    return out


def replica_log(log: str | None, k: int, name: str) -> str | None:
    """A replica's engine log: the model's own for the first card, one beside it for each other."""
    if not log or k == 0:
        return log
    p = Path(log)
    return str(p.with_name(f"{p.stem}.{name}{p.suffix}"))


class AllLocks:
    """The replicas' control lines taken together, in order; a lock-like object for Service's sleep and VRAM turns."""

    def __init__(self, locks):
        self.locks = locks

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        deadline = None if timeout is None or timeout < 0 else time.monotonic() + timeout
        taken = []
        for lock in self.locks:
            left = -1 if deadline is None else max(0.0, deadline - time.monotonic())
            if not (lock.acquire(True, left) if blocking else lock.acquire(False)):
                for t in reversed(taken):
                    t.release()
                return False
            taken.append(lock)
        return True

    def release(self) -> None:
        for lock in reversed(self.locks):
            lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


class ReplicaConsole:
    """The replicas' XSLOT figures under the combined slot numbers; the first card's expert map."""

    def __init__(self, owner: "ReplicaEngine"):
        self.owner = owner

    def slot(self, b: int, since: float | None = None) -> dict:
        r, local = self.owner.slot_home(b)
        console = getattr(r, "console", None)
        return console.slot(local, since) if console is not None else {"accept_pos": None, "accept_hist": None, "hit": None}

    def expert_map(self) -> dict | None:
        console = getattr(self.owner.replicas[0], "console", None)
        return console.expert_map() if console is not None else None


class ReplicaEngine:
    """Several ProcessEngines, one per card, as one engine."""

    def __init__(self, replicas: list, cards: list):
        self.replicas, self.cards = list(replicas), list(cards)
        n = len(self.replicas)
        self.lock = threading.Lock()                     # the routing books below
        self.active = [0] * n                            # requests running or waiting on each replica
        self.blocks = [collections.OrderedDict() for _ in range(n)]   # block hashes of the prompts each one read
        self.ends = [collections.OrderedDict() for _ in range(n)]     # the whole prompts each one read: end -> blocks
        self.sessions: collections.OrderedDict[str, int] = collections.OrderedDict()   # client session -> its replica
        self.solo = [threading.Lock() for _ in range(n)]   # a replica without batch slots runs one request at a time
        self.tl = threading.local()                      # this thread's request: .replica, .cancel, .seen, .shown
        self.console = ReplicaConsole(self)

    # -- routing --------------------------------------------------------------------------------------------------

    def room(self, i: int) -> int:
        return max(1, int(getattr(self.replicas[i], "batch", 0) or 0))

    def held(self, i: int, p: Prompt) -> int:
        """How many leading blocks of a prompt replica `i` read lately (caller holds the lock)."""
        kept, n = self.blocks[i], 0
        for h in p.hashes:
            if h not in kept:
                break
            n += 1
        return n

    def continued(self, i: int, p: Prompt) -> int:
        """The blocks of the longest earlier prompt replica `i` read that this one continues (caller holds the lock)."""
        return max((p.continues(end) for end in self.ends[i]), default=0)

    def choose(self, ids, session: str | None = None, cancel=None) -> int:
        """The replica for a prompt, booked until the request ends. A later pass of the same request (the same cancel
        event on the same thread) stays where the request runs."""
        with self.lock:
            if cancel is not None and getattr(self.tl, "cancel", None) is cancel:
                best = self.tl.replica
                self.active[best] += 1
                return best
            p = Prompt(ids)
            n = len(self.replicas)
            alive = [i for i in range(n) if self.replicas[i].alive()] or list(range(n))
            cont = {i: self.continued(i, p) for i in alive}
            home = max(alive, key=lambda i: cont[i])
            if cont[home] >= STAY_BLOCKS:
                best = home
            else:
                pool = [i for i in alive if self.active[i] < self.room(i)] or alive
                held = {i: self.held(i, p) for i in pool}
                own = self.sessions.get(session) if session is not None else None
                best = max(pool, key=lambda i: (held[i], i == own, -self.active[i] / self.room(i), -i))
            self.active[best] += 1
            self.remember(best, p, session)
            self.tl.replica, self.tl.cancel, self.tl.seen = best, cancel, WAITING
            return best

    def remember(self, i: int, p: Prompt, session: str | None) -> None:
        kept = self.blocks[i]
        for h in p.hashes:
            kept[h] = None
            kept.move_to_end(h)
        while len(kept) > BLOCKS_KEPT:
            kept.popitem(last=False)
        if p.end is not None:
            self.ends[i][p.end] = None
            self.ends[i].move_to_end(p.end)
            while len(self.ends[i]) > ENDS_KEPT:
                self.ends[i].popitem(last=False)
        if session is not None:
            self.sessions[session] = i
            self.sessions.move_to_end(session)
            while len(self.sessions) > SESSIONS_KEPT:
                self.sessions.popitem(last=False)

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None, stop_soon=None):
        i = self.choose(ids, session, cancel)
        r = self.replicas[i]
        solo = self.solo[i] if not getattr(r, "batch", 0) else None
        try:
            if solo is not None:
                while not solo.acquire(timeout=10):
                    if cancel is not None and cancel.is_set():
                        return
                    yield None
            try:
                self.tl.seen = r.last                    # a DONE replaces it: from here on, this request's
                more = {"stop_soon": stop_soon} if stop_soon is not None else {}
                yield from r.generate(ids, max_new, sampling, cancel, embeddings=embeddings, session=session, **more)
            finally:
                if solo is not None:
                    solo.release()
        finally:
            with self.lock:
                self.active[i] -= 1

    def encode_images(self, jobs: list, timeout: float | None = None) -> list[str]:
        return self.encode_on(lambda r: r.encode_images(jobs, timeout=timeout))

    def encode_video(self, video, out, words: str = "", timeout: float | None = None) -> str:
        return self.encode_on(lambda r: r.encode_video(video, out, words, timeout=timeout))

    def encode_on(self, call):
        """Run an encode on the least busy replica that can take it now: one with batch slots takes its own control
        lines, one without is held free of requests meanwhile; with none free, the first that frees."""
        while True:
            with self.lock:
                alive = [i for i in range(len(self.replicas)) if self.replicas[i].alive()] or [0]
                order = sorted(alive, key=lambda i: (self.active[i] / self.room(i), i))
            for i in order:
                r = self.replicas[i]
                if getattr(r, "batch", 0):
                    return call(r)
                if self.solo[i].acquire(False):
                    try:
                        return call(r)
                    finally:
                        self.solo[i].release()
            time.sleep(ENCODE_POLL_S)

    def mine(self):
        """The replica this thread's request runs on, None before it has one."""
        i = getattr(self.tl, "replica", None)
        return None if i is None else self.replicas[i]

    # -- slots ----------------------------------------------------------------------------------------------------

    @property
    def batch(self) -> int:
        """Every replica's slots, a replica without batch slots counting as one."""
        return sum(self.room(i) for i in range(len(self.replicas)))

    def slot_home(self, b: int):
        """The replica of combined slot `b` and its own number there."""
        for i, r in enumerate(self.replicas):
            n = self.room(i)
            if b < n:
                return r, b
            b -= n
        return self.replicas[-1], b

    def slot_view(self) -> list[dict]:
        out = []
        for i, r in enumerate(self.replicas):
            view = r.slot_view() if getattr(r, "batch", 0) else []
            out += view if len(view) == self.room(i) else \
                [{"owner": None, "held": 0} for _ in range(self.room(i))]
        return out

    @property
    def slot_drafts(self) -> dict:
        out, base = {}, 0
        for i, r in enumerate(self.replicas):
            for b, v in (getattr(r, "slot_drafts", None) or {}).items():
                out[base + b] = v
            base += self.room(i)
        return out

    def queue_depth(self) -> int:
        return sum(r.queue_depth() for r in self.replicas if getattr(r, "batch", 0) and hasattr(r, "queue_depth"))

    @property
    def ctl(self):
        locks = [getattr(r, "ctl", None) for r in self.replicas]
        return AllLocks(locks) if all(lk is not None for lk in locks) else None

    # -- this thread's request ------------------------------------------------------------------------------------

    @property
    def last(self):
        """This thread's figures of its last DONE: what was shown before, until its replica writes a new one."""
        r = self.mine()
        seen = getattr(self.tl, "seen", WAITING)
        if r is not None and seen is not WAITING and r.last is not seen:
            self.tl.seen = self.tl.shown = r.last
        if not hasattr(self.tl, "shown"):
            self.tl.shown = {}
        return self.tl.shown

    def window_stats(self):
        r = self.mine()
        return r.window_stats() if r is not None else (0, 0)

    def prompt_progress(self):
        r = self.mine()
        return r.prompt_progress() if r is not None else {}

    @property
    def progress(self):
        r = self.mine()
        if r is not None:
            return r.progress
        return next((r.progress for r in self.replicas if getattr(r, "progress", None)), None)

    @property
    def prefill_tok_s_mean(self):
        r = self.mine()
        if r is not None and r.prefill_tok_s_mean is not None:
            return r.prefill_tok_s_mean
        return next((r.prefill_tok_s_mean for r in self.replicas if r.prefill_tok_s_mean is not None), None)

    # -- facts ----------------------------------------------------------------------------------------------------

    @property
    def max_context(self) -> int:
        ctx = [int(getattr(r, "max_context", 0) or 0) for r in self.replicas]
        return min(ctx) if all(ctx) else 0

    @property
    def known_ctx(self) -> int:
        ctx = [int(getattr(r, "known_ctx", 0) or 0) for r in self.replicas]
        return min(ctx) if all(ctx) else 0

    @property
    def gen(self) -> int:
        return sum(int(getattr(r, "gen", 0) or 0) for r in self.replicas)

    @property
    def conversation(self) -> dict:
        """The cards' conversation caches: counts and sizes summed, times the slowest card's."""
        out = {}
        for r in self.replicas:
            for k, v in (getattr(r, "conversation", None) or {}).items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    out[k] = v
                else:
                    out[k] = max(out.get(k, 0), v) if k.endswith("_ms") else out.get(k, 0) + v
        return out

    @property
    def log_path(self):
        """The log of the first card whose engine ended, else the first card's."""
        return next((r.log_path for r in self.replicas if not r.alive() and r.log_path), self.replicas[0].log_path)

    def gpu_state_mib(self) -> int | None:
        state = [r.gpu_state_mib() for r in self.replicas]
        return None if any(s is None for s in state) else sum(state)

    def __getattr__(self, name):
        """What every replica has alike (the model file, its facts, the encoder's state): the first card's."""
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.replicas[0], name)

    # -- lifecycle ------------------------------------------------------------------------------------------------

    def alive(self) -> bool:
        return all(r.alive() for r in self.replicas)

    @property
    def asleep(self) -> bool:
        return any(getattr(r, "asleep", False) is True for r in self.replicas)

    @property
    def unloaded(self) -> bool:
        return all(getattr(r, "unloaded", False) for r in self.replicas)

    @property
    def starting(self) -> bool:
        return any(getattr(r, "starting", False) for r in self.replicas)

    @property
    def loading(self) -> bool:
        return any(getattr(r, "loading", False) for r in self.replicas)

    @property
    def load_status(self) -> dict:
        """The load furthest behind."""
        loads = [getattr(r, "load_status", None) or {} for r in self.replicas if getattr(r, "loading", False)]
        return min(loads, key=lambda s: s.get("percent") or 0) if loads else {}

    @load_status.setter
    def load_status(self, value: dict) -> None:
        for r in self.replicas:
            r.load_status = dict(value)

    def restart(self, tries: int = 3):
        """Start every replica that is not running, one after another: each sizes its RAM after the ones before."""
        down = [i for i, r in enumerate(self.replicas) if not r.alive()] or list(range(len(self.replicas)))
        with self.lock:                                 # a new process holds none of the conversations
            for i in down:
                self.blocks[i].clear()
                self.ends[i].clear()
        for i in down:
            self.replicas[i].restart(tries=tries)

    def each(self, name: str, *args, **kwargs) -> list:
        """Call every replica at once; the first failure is raised once all have answered."""
        out, errors = [None] * len(self.replicas), []

        def run(k, r):
            try:
                out[k] = getattr(r, name)(*args, **kwargs)
            except Exception as e:                      # noqa: BLE001 - raised below
                errors.append(e)
        threads = [threading.Thread(target=run, args=(k, r), daemon=True) for k, r in enumerate(self.replicas)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return out

    def unload(self):
        self.each("unload")

    def discard(self):
        self.each("discard")

    def close(self):
        self.each("close")

    def reap(self):
        self.each("reap")

    def can_sleep(self) -> bool:
        return all(r.can_sleep() for r in self.replicas)

    def sleep(self, ram_check=None, timeout: float = 120.0) -> dict:
        """Sleep every replica, one after another, each checking the RAM the ones before left; one that will not sleep
        wakes the others again and raises its reason."""
        done, out = [], {}
        for r in self.replicas:
            try:
                out = r.sleep(ram_check=ram_check, timeout=timeout)
            except ValueError:
                for d in reversed(done):
                    with contextlib.suppress(Exception):   # the refusal is the answer; a failed wake ends that engine
                        d.wake(timeout=timeout)
                raise
            done.append(r)
        return out

    def wake(self, timeout: float = 120.0, vram_check=None) -> dict:
        out = {}
        for r in self.replicas:
            if getattr(r, "asleep", False) is True:
                out = r.wake(timeout=timeout, vram_check=vram_check)
        return out

    @property
    def restore_s(self) -> float:
        return sum(float(getattr(r, "restore_s", 0.0) or 0.0) for r in self.replicas)

    def vram(self, reserve_mib, timeout: float = 120.0) -> dict:
        return self.each("vram", reserve_mib, timeout)[0]

    def held_mib(self) -> int | None:
        held = [r.held_mib() for r in self.replicas]
        return None if any(h is None for h in held) else sum(held)

    def death_note(self) -> str:
        return "; ".join(f"GPU {c}: {r.death_note()}" for c, r in zip(self.cards, self.replicas) if not r.alive())

    def exit_code(self):
        return next((r.exit_code() for r in self.replicas if not r.alive()), None)
