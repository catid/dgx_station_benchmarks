# MiniMax H3 duration-scaling data

All rows use official BF16 FL2VA weights, 1344×768, 24 FPS, 50 sigma-grid
points / 49 model evaluations, video/audio flow shifts 12/3, one output, and no
offload. Client time covers asynchronous submission through completed MP4
download.

| Workload | Runtime state | Replicas | Client E2E | Peak HBM | Encoded output | Clips/hour |
|---|---|---:|---:|---:|---:|---:|
| Canonical 5-second VBench | warmed; mean of 3 | 1 | 116.865 s | 125,268 MB | 124 frames / 5.167 s | 30.805 |
| Rocket cat 10-second sample | warmed 10-second shape | 1 | 334.731 s | 131,946 MB | 243 frames / 10.125 s | 10.755 |
| Mountain cat 15-second sample | cold first 15-second shape | 2 independent | 719.290 s mean | 141,424 MB | 362 frames / 15.083 s | 5.005 each |
| Same simultaneous 15-second pair | cold first 15-second shape | 2 independent | 719.296 s pair makespan | 141,424 MB each | 724 aggregate frames | 10.010 aggregate |

The two 15-second station results were submitted at the same time with the
same prompt and seed. Client end-to-end times were 719.296342 and 719.283174
seconds (0.013167-second range); the MP4s were byte-identical at SHA-256
`bb14ce9710f03fcbf2464a28431afabab61b901a2134fbee8e1fd0c538929efd`.

## Quality notes

- The selected rocket seed 1011 keeps the spacesuited cat visible and securely
  astride the rocket from launch through orbit. Motion and background evolve;
  no repetition or collapse was seen.
- An earlier rocket seed 1010 produced a coherent launch-to-orbit sequence but
  omitted the cat. It is excluded from the gallery as a semantic prompt miss,
  not a decode or degeneracy failure. Its request, normalized metrics, and
  contact sheet are retained in `rocket-cat-10s-v1-semantic-miss/`.
- The mountain sample keeps the goggled cat and terrain coherent through
  multiple tracking shots and a stylized long leap; no impact or injury is
  shown. It has active motion throughout rather than repeated/frozen frames.
- Selected outputs fully decoded with H.264 video and 32-kHz stereo AAC. The
  validator reported no black interval ≥0.5 seconds, freeze ≥1 second, or
  silence ≥0.25 seconds below −50 dB.

The per-run subdirectories retain request, timing, job, `ffprobe`, warning,
hash, and normalized summary records. The compact cross-row CSV and generated
charts are added after the single experimental 30-second attempt completes.
