"""gpu-mode replica: one engine per card behind one Service; a conversation stays on the replica that read it.

The fake batch engine of test_parallel runs behind each real ProcessEngine."""
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.engine.process import ProcessEngine
from serve.engine.replicas import BLOCK, STAY_BLOCKS, AllLocks, Prompt, ReplicaEngine, replica_args, replica_log
from serve.engine.vision import check_gpu, replica_cards
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.tests.test_parallel import FAKE_BATCH
from serve.util.text import ByteTokenizer


class Stub:
    """A replica as routing sees it."""

    def __init__(self, batch=2, up=True):
        self.batch, self.up, self.max_context = batch, up, 4096

    def alive(self):
        return self.up


class Routing(unittest.TestCase):
    def test_equal_starts_give_equal_hashes_whole_blocks_only(self):
        a = list(range(BLOCK * 3 + 5))
        b = a[:BLOCK * 2] + [-1] * (BLOCK + 5)
        ha, hb = Prompt(a).hashes, Prompt(b).hashes
        self.assertEqual(len(ha), 3)
        self.assertEqual(ha[:2], hb[:2])
        self.assertNotEqual(ha[2], hb[2])
        self.assertEqual(Prompt(a[:BLOCK - 1]).hashes, [])

    def test_a_prompt_continues_an_earlier_one_only_when_it_holds_all_of_it(self):
        first = list(range(BLOCK * 3 + 5))
        self.assertEqual(Prompt(first + [9, 9]).continues(Prompt(first).end), 3)
        self.assertEqual(Prompt(first[:-1] + [9, 9]).continues(Prompt(first).end), 0)

    def test_a_conversation_goes_back_to_its_replica(self):
        e = ReplicaEngine([Stub(), Stub()], [0, 1])
        first = list(range(BLOCK * 4))
        other = [7] * (BLOCK * 4)
        i = e.choose(first)
        j = e.choose(other)
        self.assertNotEqual(i, j)                        # a new prompt goes to the least busy replica
        e.active = [0, 0]
        self.assertEqual(e.choose(first + [1, 2, 3]), i)
        self.assertEqual(e.choose(other + [4]), j)

    def test_a_full_replica_gives_a_short_conversation_to_one_with_room(self):
        e = ReplicaEngine([Stub(batch=1), Stub(batch=1)], [0, 1])
        conv = list(range(BLOCK * 2))
        i = e.choose(conv)                               # still running: its replica is full
        self.assertEqual(e.choose(conv + [5]), 1 - i)

    def test_a_long_conversation_waits_for_its_own_replica(self):
        e = ReplicaEngine([Stub(batch=1), Stub(batch=1)], [0, 1])
        conv = list(range(BLOCK * STAY_BLOCKS + 3))
        i = e.choose(conv)                               # still running: its replica is full
        self.assertEqual(e.choose(conv + [5]), i)

    def test_a_shared_system_prompt_pins_no_new_conversation(self):
        e = ReplicaEngine([Stub(batch=1), Stub(batch=1)], [0, 1])
        system = list(range(BLOCK * STAY_BLOCKS * 2))
        i = e.choose(system + [1, 2, 3])                 # one agent runs; another starts with the same system prompt
        self.assertEqual(e.choose(system + [4, 5, 6]), 1 - i)

    def test_a_later_pass_of_a_request_stays_on_its_replica(self):
        e = ReplicaEngine([Stub(), Stub()], [0, 1])
        cancel = threading.Event()
        i = e.choose([1] * BLOCK, cancel=cancel)
        e.active[i] += 5                                 # the other replica is now far less busy
        self.assertEqual(e.choose([2] * BLOCK, cancel=cancel), i)
        self.assertEqual(e.choose([2] * BLOCK, cancel=threading.Event()), 1 - i)

    def test_a_restarted_replica_holds_no_conversation(self):
        class Dead(Stub):
            def restart(self, tries=3):
                self.up = True
        e = ReplicaEngine([Dead(), Stub()], [0, 1])
        conv = list(range(BLOCK * STAY_BLOCKS))
        e.choose(conv)
        e.replicas[0].up = False
        e.restart()
        self.assertEqual(e.held(0, Prompt(conv)), 0)
        self.assertEqual(e.continued(0, Prompt(conv + [1])), 0)

    def test_last_is_this_requests_done_only(self):
        e = ReplicaEngine([Stub(), Stub()], [0, 1])
        for r in e.replicas:
            r.last = {"from": "an earlier request"}
        before = e.last
        e.choose([1] * BLOCK)
        self.assertIs(e.last, before)                    # waiting for its replica: nothing new
        e.tl.seen = e.mine().last                        # it holds the replica (as generate does)
        self.assertIs(e.last, before)                    # no DONE yet
        e.mine().last = {"from": "this request"}
        self.assertEqual(e.last, {"from": "this request"})

    def test_a_dead_replica_takes_nothing_while_another_lives(self):
        e = ReplicaEngine([Stub(up=False), Stub()], [0, 1])
        self.assertEqual({e.choose([n] * BLOCK) for n in range(3)}, {1})

    def test_the_session_breaks_a_tie(self):
        e = ReplicaEngine([Stub(), Stub()], [0, 1])
        e.active = [0, 1]
        e.sessions["agent"] = 1
        self.assertEqual(e.choose([1, 2], session="agent"), 1)
        self.assertEqual(e.choose([1, 2]), 0)

    def test_text_prompts_route_by_their_text(self):
        class Text:
            def __init__(self, text):
                self.text = text
        e = ReplicaEngine([Stub(), Stub()], [0, 1])
        body = "x" * (BLOCK * 8)
        i = e.choose(Text(body))
        e.choose(Text("y" * (BLOCK * 8)))
        e.active = [0, 0]
        self.assertEqual(e.choose(Text(body + " next turn")), i)

    def test_logit_bias_follows_each_cards_own_slots(self):
        from serve.api.logit_bias import checked
        req = {"logit_bias": {"5": 10}}
        self.assertEqual(checked(req, ReplicaEngine([Stub(batch=0), Stub(batch=0)], [0, 1])), {5: 10.0})
        with self.assertRaisesRegex(ValueError, "parallel slots"):
            checked(req, ReplicaEngine([Stub(batch=2), Stub(batch=2)], [0, 1]))

    def test_slots_are_numbered_card_after_card(self):
        e = ReplicaEngine([Stub(batch=2), Stub(batch=0)], [0, 1])
        self.assertEqual(e.batch, 3)
        self.assertEqual([e.slot_home(b)[1] for b in range(3)], [0, 1, 0])
        self.assertIs(e.slot_home(2)[0], e.replicas[1])


class Settings(unittest.TestCase):
    def test_the_mode_is_a_gpu_mode_without_shares(self):
        check_gpu("0,1", "replica")
        check_gpu("all", "replica")
        with self.assertRaisesRegex(ValueError, "shares do not apply"):
            check_gpu("0,1", "replica", "3,1")
        with self.assertRaisesRegex(ValueError, "a mode needs two or more"):
            check_gpu("0", "replica")

    def test_all_is_every_visible_card(self):
        self.assertEqual(replica_cards("1,0"), ["1", "0"])
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "3,5"}):
            self.assertEqual(replica_cards("all"), ["0", "1"])

    def test_each_replica_parks_in_its_own_folder_on_its_part_of_the_budgets(self):
        args = ["--model", "m", "--conversation-cache-mib", "64000", "--conversation-cache-disk-mib", "1001",
                "--conversation-cache-dir", str(Path("c"))]
        out = replica_args(args, "gpu3", 2)
        self.assertEqual(out[out.index("--conversation-cache-mib") + 1], "32000")
        self.assertEqual(out[out.index("--conversation-cache-disk-mib") + 1], "500")
        self.assertEqual(out[out.index("--conversation-cache-dir") + 1], str(Path("c") / "gpu3"))
        self.assertEqual(replica_args(["--model", "m"], "gpu1", 2), ["--model", "m"])
        tiny = replica_args(["--conversation-cache-mib", "1", "--conversation-cache-disk-mib", "0"], "gpu1", 2)
        self.assertEqual(tiny, ["--conversation-cache-mib", "1", "--conversation-cache-disk-mib", "0"])

    def test_the_first_card_keeps_the_model_log(self):
        log = str(Path("d") / "engine.log")
        self.assertEqual(replica_log(log, 0, "gpu0"), log)
        self.assertEqual(replica_log(log, 1, "gpu1"), str(Path("d") / "engine.gpu1.log"))
        self.assertIsNone(replica_log(None, 1, "gpu1"))

    def test_all_locks_take_none_when_one_is_held(self):
        a, b = threading.Lock(), threading.Lock()
        b.acquire()
        both = AllLocks([a, b])
        self.assertFalse(both.acquire(blocking=False))
        self.assertFalse(a.locked())
        b.release()
        self.assertTrue(both.acquire(blocking=False))
        self.assertTrue(a.locked() and b.locked())
        both.release()
        self.assertFalse(a.locked() or b.locked())


class ReplicaService(unittest.TestCase):
    """Two real ProcessEngines with the fake engine behind one Service over HTTP."""

    def start(self, slots=2):
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_engine.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        self.logs = [Path(self.tmp.name) / f"requests{k}.log" for k in range(2)]
        real = subprocess.Popen
        fake = mock.patch.object(subprocess, "Popen", lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        fake.start()                                    # a restart starts the fake again
        self.addCleanup(fake.stop)
        parts = [ProcessEngine("basalt", (["--batch", str(slots)] if slots else []) + ["--ready", "65536", "--log", str(log)])
                 for log in self.logs]
        self.engine = ReplicaEngine(parts, [0, 1])
        self.svc = Service(self.engine, ByteTokenizer(),
                           ChatTemplate.from_file(Path(__file__).resolve().parents[1] / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def chat(self, messages, max_tokens=64):
        body = {"model": "basalt", "messages": messages, "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    def served(self) -> list[list[int]]:
        """The prompt lengths of the requests each replica's fake engine ran (its GEN and BGEN lines)."""
        return [[int(x.split()[2]) for x in log.read_text().splitlines() if x.startswith(("GEN", "BGEN"))]
                if log.exists() else [] for log in self.logs]

    def conversations(self, slots):
        self.start(slots)
        self.assertEqual(self.engine.batch, 2 * max(1, slots))
        doc = {"A": "conversation A: " + "lorem ipsum " * 100, "B": "conversation B: " + "dolor sit amet " * 150}
        errors = []

        def turn(k, n):
            try:
                ask = f"turn {n}" + (" LONGREPLY" if n == 0 else "")
                self.chat([{"role": "system", "content": doc[k]}, {"role": "user", "content": ask}])
            except Exception as e:  # noqa: BLE001 - reported below
                errors.append(e)
        threads = [threading.Thread(target=turn, args=(k, 0)) for k in "AB"]
        for t in threads:
            t.start()
            time.sleep(0.05)
        for t in threads:
            t.join(30)
        for k in "AABB":                                 # later turns, one at a time
            turn(k, 1)
        self.assertEqual(errors, [])
        # two at once went one to each card, and every later turn to the card that read its conversation (B's prompts
        # are the longer ones)
        long_ones = [sorted(n > 2000 for n in lens) for lens in self.served()]
        self.assertEqual(sorted(long_ones), [[False] * 3, [True] * 3])
        self.assertEqual(self.engine.active, [0, 0])

    def test_conversations_stay_on_their_cards_with_batch_slots(self):
        self.conversations(2)

    def test_conversations_stay_on_their_cards_with_one_slot_each(self):
        self.conversations(0)

    def test_slot_rows_cover_both_cards(self):
        self.start(2)
        self.chat([{"role": "user", "content": "hello"}])
        rows, queue = self.svc.slot_table()
        self.assertEqual([r["id"] for r in rows], [0, 1, 2, 3])
        self.assertEqual(queue, [])

    def test_a_replica_that_ended_is_started_again_alone(self):
        self.start(2)
        self.chat([{"role": "user", "content": "hello"}])
        first, second = self.engine.replicas
        gen1 = second.gen
        first.proc.kill()
        first.proc.wait(10)
        self.assertFalse(self.engine.alive())
        self.assertIn("GPU 0", self.engine.death_note())
        self.assertEqual(self.chat([{"role": "user", "content": "again"}]), "ok, done.")
        self.assertTrue(self.engine.alive())
        self.assertEqual(second.gen, gen1)                # the live one was left running


if __name__ == "__main__":
    unittest.main()
