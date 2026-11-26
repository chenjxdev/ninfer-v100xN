# Checking the first-generation DFlash draft (MMFlash)

The draft's correctness is hard to judge from the acceptance rate alone: a
wrong draft looks exactly like a rejected one.  These tools compare the
engine's draft against the reference implementation, stage by stage.

## The engine's trace

`NINFER_MMFLASH_TRACE=<prefix>` makes the engine write the draft's inputs and
outputs next to the running service:

| file | what it holds |
| --- | --- |
| `pending_features` | the target features the capture wrote, before they are compacted |
| `context_features` | the compacted features the draft's context is built from |
| `context_projected` | `hidden_norm(fc(features))`, the draft's context |
| `proposal_ids`, `proposal_positions` | the block the draft proposes |
| `block_embedding` | the target's embedding of the block |
| `draft_hidden` | the draft's hidden state |
| `draft_tokens` | the tokens the draft proposed |

The trace stays outside a CUDA graph capture, so the service has to run with
`--no-cuda-graph` for the files to appear.

## The tools

| script | what it answers |
| --- | --- |
| `check_artifact.py` | is the draft package in the artifact intact? |
| `repair_inplace.py` | repairs a draft package whose payloads are shifted |
| `compare4.py` | does the engine's context projection and draft match the reference? |
| `find_taps.py` | which of the five target taps carry data? |

They need the reference implementation in a virtualenv with the `dflash`
package, `transformers` and `torch`:

```bash
pip install 'torch==2.13.0' 'transformers==5.15.0' dflash
```

## The open problem

The engine captures the target's hidden states at five layers, and the draft
reads them as one concatenated vector.  In the decode phase only the first of
the five carries data; the other four are zero, which is why the acceptance
rate sits at 5.3 percent instead of the range the reference reports.

What has been checked so far:

- the layer numbers are right: layer 1 fills tap 0, 16 fills tap 1, 31 fills
  tap 2, 46 fills tap 3, 61 fills tap 4
- the captured layer outputs are not empty: every source the copy reads holds
  non-zero values
- the slot each tap writes to has the right address, spaced 5120 elements apart
- the compaction and the scatter walk their destinations with the strides the
  layout gives them

What has not: where the four taps lose their data between the copy and the
compaction.  `find_taps.py` reads the pending buffer per tap; run it with the
trace enabled to see where the zeros appear.
