"""gpu-mode replica: one engine process per card, each running the whole model, behind the one engine a Service drives.

A request goes to the replica whose recent prompts share the longest start with it, so a conversation keeps reading only
its new tokens on the card that holds the rest; among the replicas with room, else all of them. A prompt no replica
holds goes to the least busy one. Every lifecycle call (start, sleep, wake, VRAM, unload) applies to every replica; the
slots are numbered card after card.
"""
from __future__ import annotations

import collections
import contextlib
import threading
from pathlib import Path

from serve.engine.flags import flag_index, flag_value

BLOCK = 256              # prompt ids per routing block (characters for a text prompt: four times as many)
BLOCKS_KEPT = 1 << 18    # block hashes remembered per replica, least recently used dropped first
SESSIONS_KEPT = 4096     # client sessions remembered


def block_hashes(ids) -> list[int]:
    """One hash per whole block of a prompt (its ids, or the rendered text the engine tokenizes), each covering the
    blocks before it too: equal hashes mean an equal start."""
    text = getattr(ids, "text", None)
    seq, size = (text, BLOCK * 4) if isinstance(text, str) else (ids, BLOCK)
    out, h = [], 0
    for i in range(0, len(seq) - size + 1, size):
        part = seq[i:i + size]
        h = hash((h, part if isinstance(part, str) else tuple(part)))
        out.append(h)
    return out


SHARED_BUDGETS = ("--conversation-cache-mib", "--conversation-cache-disk-mib")   # the machine's RAM and disk, split


def replica_args(args: list[str], card, count: int) -> list[str]:
    """A replica's engine arguments: its own folder for the conversations it parks on disk, and its part of the RAM
    and disk budgets the replicas share."""
    out = list(args)
    where = flag_value(out, "--conversation-cache-dir")
    if where:
        out[flag_index(out, "--conversation-cache-dir") + 1] = str(Path(where) / f"gpu{card}")
    for flag in SHARED_BUDGETS:
        value = flag_value(out, flag)
        if value and value.isdigit():
            out[flag_index(out, flag) + 1] = str(int(value) // count)
    return out


def replica_log(log: str | None, k: int, card) -> str | None:
    """A replica's engine log: the model's own for the first card, one beside it for each other."""
    if not log or k == 0:
        return log
    p = Path(log)
    return str(p.with_name(f"{p.stem}.gpu{card}{p.suffix}"))


class AllLocks:
    """The replicas' control lines taken together, in order; a lock-like object for Service's sleep and VRAM turns."""

    def __init__(self, locks):
        self.locks = locks

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        taken = []
        for lock in self.locks:
            if not (lock.acquire(blocking, timeout) if blocking else lock.acquire(False)):
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

    def __init__(self, replicas: list, cards: list[int]):
        self.replicas, self.cards = list(replicas), list(cards)
        self.lock = threading.Lock()                     # the routing books below
        self.active = [0] * len(self.replicas)           # requests running or waiting on each replica
        self.blocks = [collections.OrderedDict() for _ in self.replicas]   # block hashes of the prompts each one read
        self.sessions: collections.OrderedDict[str, int] = collections.OrderedDict()   # client session -> its replica
        self.solo = [threading.Lock() for _ in self.replicas]   # a replica without batch slots runs one request at a time
        self.tl = threading.local()                      # .replica: the one this thread's request runs on
        self.console = ReplicaConsole(self)

    # -- routing --------------------------------------------------------------------------------------------------

    def room(self, i: int) -> int:
        return max(1, int(getattr(self.replicas[i], "batch", 0) or 0))

    def held(self, i: int, hashes: list[int]) -> int:
        """How many leading blocks of a prompt replica `i` read lately (caller holds the lock)."""
        kept, n = self.blocks[i], 0
        for h in hashes:
            if h not in kept:
                break
            n += 1
        return n

    def choose(self, ids, session: str | None = None) -> int:
        """The replica for a prompt: the longest held start among those with a free slot (else among all), then the
        client session's own, then the least busy; the choice is booked until the request ends."""
        hashes = block_hashes(ids)
        with self.lock:
            n = len(self.replicas)
            alive = [i for i in range(n) if self.replicas[i].alive()] or list(range(n))
            pool = [i for i in alive if self.active[i] < self.room(i)] or alive
            held = {i: self.held(i, hashes) for i in pool}
            own = self.sessions.get(session) if session is not None else None
            best = max(pool, key=lambda i: (held[i], i == own, -self.active[i] / self.room(i), -i))
            self.active[best] += 1
            kept = self.blocks[best]
            for h in hashes:
                kept[h] = None
                kept.move_to_end(h)
            while len(kept) > BLOCKS_KEPT:
                kept.popitem(last=False)
            if session is not None:
                self.sessions[session] = best
                self.sessions.move_to_end(session)
                while len(self.sessions) > SESSIONS_KEPT:
                    self.sessions.popitem(last=False)
            return best

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None, stop_soon=None):
        i = self.choose(ids, session)
        self.tl.replica = i
        r = self.replicas[i]
        solo = self.solo[i] if not getattr(r, "batch", 0) else None
        try:
            if solo is not None:
                while not solo.acquire(timeout=10):
                    if cancel is not None and cancel.is_set():
                        return
                    yield None
            try:
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
        """Run an encode on the least busy replica; one without batch slots is free of requests meanwhile (a replica
        with batch slots takes its own control lines)."""
        with self.lock:
            alive = [i for i in range(len(self.replicas)) if self.replicas[i].alive()] or [0]
            i = min(alive, key=lambda i: (self.active[i] / self.room(i), i))
        r = self.replicas[i]
        with self.solo[i] if not getattr(r, "batch", 0) else contextlib.nullcontext():
            return call(r)

    def mine(self):
        """The replica this thread's request runs on (the first before it has one)."""
        return self.replicas[getattr(self.tl, "replica", 0)]

    # -- slots ----------------------------------------------------------------------------------------------------

    @property
    def batch(self) -> int:
        """Every replica's slots, a replica without batch slots counting as one; 0 while none is up."""
        if not any(getattr(r, "max_context", 0) for r in self.replicas):
            return 0
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
        return self.mine().last if hasattr(self.tl, "replica") else {}

    def window_stats(self):
        return self.mine().window_stats()

    def prompt_progress(self):
        return self.mine().prompt_progress()

    @property
    def progress(self):
        return next((r.progress for r in self.replicas if getattr(r, "progress", None)), None)

    @property
    def prefill_tok_s_mean(self):
        return self.mine().prefill_tok_s_mean

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
        out = {}
        for r in self.replicas:
            for k, v in (getattr(r, "conversation", None) or {}).items():
                out[k] = out.get(k, 0) + v if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        return out

    @property
    def proc(self):
        """None: the replicas' processes are several, so the sleep checks read the cards' memory instead."""
        return None

    def __getattr__(self, name):
        """What every replica has alike (the model file, its facts, the encoder): the first card's."""
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
        """Start every replica that is not running, at once; the first failure is raised once all have ended."""
        down = [r for r in self.replicas if not r.alive()] or self.replicas
        errors: list[BaseException] = []

        def start(r):
            try:
                r.restart(tries=tries)
            except BaseException as e:                  # noqa: BLE001 - raised below
                errors.append(e)
        threads = [threading.Thread(target=start, args=(r,), daemon=True) for r in down]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]

    def each(self, name: str, *args, **kwargs) -> list:
        return [getattr(r, name)(*args, **kwargs) for r in self.replicas if hasattr(r, name)]

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
        """Sleep every replica; one that will not sleep wakes the others again and raises its reason."""
        done, out = [], {}
        for r in self.replicas:
            try:
                out = r.sleep(ram_check=ram_check, timeout=timeout)
            except ValueError:
                for d in reversed(done):
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
        outs = self.each("vram", reserve_mib, timeout)
        return outs[0] if outs else {}

    def held_mib(self) -> int | None:
        held = [r.held_mib() for r in self.replicas]
        return None if any(h is None for h in held) else sum(held)

    def death_note(self) -> str:
        notes = [f"GPU {c}: {r.death_note()}" for c, r in zip(self.cards, self.replicas) if not r.alive()]
        return "; ".join(notes)

    def exit_code(self):
        return next((r.exit_code() for r in self.replicas if not r.alive()), None)
