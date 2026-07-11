# A Groovy Kind of Love (Phil Collins) — benchmark case

- Video: https://www.youtube.com/watch?v=WTxmmqbHe_M
  ("A Groovy Kind Of Love - Phil Collins - Piano Cover + Sheet Music")
- Source fingering.json: SMaPE output from 2026-07-08, copied from
  `~/Downloads/A Groovy Kind Of Love - Phil Collins - Piano Cover + Sheet Music.fingering.json`.
- `truth.json` was scaffolded with `benchmark.py init` from that same
  fingering.json, so right now it is **just SMaPE's own guesses**, not a
  verified ground truth.
- `predicted.json` is an unmodified copy of the same fingering.json.

## TODO (human labeling pass required)

`truth.json` still needs a human to watch the video and **correct the
`hand` field for any note where SMaPE guessed wrong** (and ideally
`finger` too, for finger-accuracy to mean anything). Until that
correction pass happens, running the benchmark on this case will show
~100% hand accuracy, which is a trivial artifact of truth == predicted,
not a real measurement.
