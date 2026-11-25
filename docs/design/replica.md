# Replica mode

How `gpu-mode replica` works, for server contributors. What users set and what they get is in
[GPUs](../configuration/gpus.md#replica-mode).

## The idea

A model that fits one card gains nothing from `peer` or `split` when there are two: one card already holds every weight, and
a split only adds crossings. What a second card can add is throughput across conversations. Replica mode runs the whole model
once per card and puts one address in front, so an agent framework that sends every request to one endpoint uses both cards
and keeps each conversation's prompt cache.

The mode lives in the server, and the engine is unchanged: each card runs an ordinary single-card engine. Routing whole
requests is the server's job ("the engine does the work, the server serves"), every model runs the mode without a line of
its own, and one conversation decodes exactly as on one card. The engine refuses the name (`gpu-mode replica: the server runs
it`), so `basalt`, `basalt-bench` and `basalt-probe` do too.

## The parts

| Part | File | What it does |
| --- | --- | --- |
| Settings | `serve/engine/vision.py` (`check_gpu`, `replica_cards`) | `replica` is a gpu-mode; `gpu-share` is refused with it; `all` is every visible card |
| Plan | `serve/server.py` (`check_model`) | the engine's `--gpu-plan` once per card, alone and without a mode; every card's notes and warnings |
| Engines | `serve/server.py` (`build_model`), `serve/engine/replicas.py` (`replica_args`, `replica_log`) | one lazy `ProcessEngine` per card, its `CUDA_VISIBLE_DEVICES` that card, `--gpu 0`, its own log and conversation folder, its part of the RAM and disk budgets |
| One engine | `serve/engine/replicas.py` (`ReplicaEngine`) | what `Service` drives: routing, the slots card after card, and every lifecycle call on every card |

`Service`, `Router` and the handler see one engine. `ReplicaEngine` implements what they call:

- **Requests:** `generate` routes, then runs the chosen card's own `generate`. A card without batch slots (`parallel 1`) runs
  one request at a time behind its own lock, so `batch` counts it as one slot and the cards still run at once.
- **This thread's request:** `last`, `window_stats`, `prompt_progress`, `progress` and `prefill_tok_s_mean` are the chosen
  card's. `last` changes only on that card's DONE after the request holds the card, so a request never takes another's
  figures as its own.
- **Slots:** `slot_view`, `slot_drafts` and the console's XSLOT figures number the slots card after card; `queue_depth` sums.
- **Facts:** the smallest `max_context`; summed `gen`, `held_mib`, `gpu_state_mib` and conversation counts; the first card's
  `info`, model path and expert map; the log of a card that ended.
- **Locks:** `ctl` takes every card's control lines in card order, for the sleep and VRAM turns.
- **Per card:** `api/logit_bias.py` asks each card whether it runs batch slots (`replicas`), not the summed `batch`.

## Routing

Each card remembers the prompts it read: chained hashes of 256-token blocks (1,024 characters for a prompt the engine
tokenizes), so equal hashes mean an equal start, and each whole prompt's end (its last block and the hash of the rest).

1. A later pass of the same request (the same cancel event on the same thread: a thinking budget, answer now, a forced call)
   stays on its card.
2. A turn that continues in full an earlier prompt of 16 blocks or more waits for that prompt's card. A shared system prompt
   is the start of many prompts but the whole of none, so it pins nothing.
3. Otherwise, among the cards with a free slot (else all): the most leading blocks held, then the card of the client's session
   header, then the least busy.

A card that starts again forgets what it read: its new process holds none of it. The blocks and the 16-block rule were
measured with three 60K-token conversations on two one-slot cards (warm servers): waiting instead of reading the whole prompt again on the
free card reused 642K prompt tokens against 535K, for the same wall time.

## Lifecycle

- **Start:** the cards load one after another, so each sizes its RAM tiers after the ones before; a restart starts only the
  cards that ended.
- **Sleep, wake:** one card after another; a card that refuses wakes the ones already asleep and the refusal is the answer.
- **Unload, close, VRAM:** every card at once; a failure is raised once all have answered.
- **Pictures:** the least busy card that can take an encode now encodes it; a card without batch slots is held free of
  requests meanwhile.
- **Budgets:** `conversation-cache-mib` and `conversation-cache-disk-mib` are divided between the cards; each parks in
  `gpu<N>` under the cache folder and logs to `engine.gpu<N>.log` beside the first card's log.

## Limits

- Every card holds the whole model: a model whose experts spill holds a RAM copy per card.
- A conversation that moves to another card reads its prompt again there; the cards share no conversation cache.
- One conversation runs on one card at the speed of one card; the mode adds throughput, not single-stream speed.

## Checks

- `serve/tests/test_replicas.py`: the routing rules with stub cards, the settings, and two real `ProcessEngine`s with the
  fake engine behind one `Service` (conversations kept on their cards with and without batch slots, the slot table, a card
  started again alone).
- Greedy identity through the server: three prompts of two turns each at temperature 0, five runs on one card and five in
  replica mode. Every reply is one of the replies one card gives; one reply has two variants on one card alone too (two of
  its five runs), and replica mode gives the same two.
