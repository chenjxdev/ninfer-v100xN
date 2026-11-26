# Comparing the speculative drafters on one box

`bench_draft_compare.sh` measures a running service with the same prompt and
the same length as the earlier MTP A/B runs, so the numbers are comparable:
a 600-character Chinese explanation of KV cache, 900 generated tokens, two
warm-up requests and three measured ones.

```bash
PORT=18099 LOG=/data/models/logs/heretic-dflash2.log \
  tools/v100/bench/bench_draft_compare.sh
```

It prints the engine's own line for each measured request, which carries the
draft acceptance and the committed decode rate:

```
[req 2] done ... decode=106.6tok/s ... speculative=mtp 2.56tok/round (39.0%)
```

`ask.sh` asks one question and prints the answer with its usage.

## The measured yardstick (2026-11-26, 2 x V100-SXM2 16 GB)

| draft | acceptance | decode |
| --- | --- | ---: |
| none | — | 51.3 tok/s |
| MTP, 4 drafts | 2.56 tok/round (39.0%) | 104–107 tok/s |
| MTP, 5 drafts | 2.63 tok/round (32.6%) | 95–102 tok/s |

MTP's draft window stops at 5, and a wider window is slower: the wider block
loses more of its proposals.  A wider block therefore has to come from a
different drafter -- DFlash2, whose package the engine already supports.
