# Experimental native-30-second audit

The pinned SGLang runtime officially accepts MiniMax H3 durations only in the
4–15-second range. An otherwise canonical 30-second request returned HTTP 400
in 0.002785 seconds with the response retained in `stock-response.json`.

Static inspection of pinned SGLang source `c0b6474b43363c2f4bc60fe3d7817d393fb51d32`
found one duration ceiling and no second fixed temporal shape:

- 30 requested seconds align from 720 to 736 output frames at 24 FPS;
- video latent time is 217 and video token rows are 218,736 at 1344×768;
- RoPE positions, attention sequence partitions, and VAE temporal chunks are
  constructed dynamically;
- the source overlay in `../../recipes/experimental-native30.patch` changes
  only `MINIMAX_H3_MAX_DURATION_SECONDS` from 15 to 30.

Source provenance:

| Artifact | SHA-256 |
|---|---|
| Stock `constants.py` | `583e94f42d36a4fb0dce06162ba8f62a9ab4045458989c15253cc341fe88cabf` |
| Patched `constants.py` | `e510e78c0355c2aec0a4b5640045e46e36dafffc20694a5c497b882bdd066cef` |
| Published path-sanitized patch | `73acef5eff8c6214c777db0dc0786848a9081c1f709e0cbed2d5479ff53c2d9a` |

This is an unsupported experiment, not a claim that upstream supports native
30-second generation. The request was limited to one attempt after a measured
15-second peak-HBM gate. It completed successfully without a retry.

## Measured result

| Metric | Exact value |
|---|---:|
| Client end to end | 2,454.441510 s |
| Server inference | 2,446.301930 s |
| Text encode | 0.0609 s |
| Latent preparation | 0.1554 s |
| Denoise | 2,365.0206 s |
| Decode | 28.9604 s |
| Peak HBM | 156,852 MB |
| Encoded video | 736 frames / 30.666667 s / 24 FPS |
| Encoded audio | AAC stereo / 32 kHz / 30.675 s |
| Frames per wall-second | 0.299864550 |
| Video-seconds per wall-second | 0.012494356 |
| Real-time factor | 80.036135× |
| Complete clips/hour | 1.466728779 |
| MP4 size | 6,608,480 bytes |
| MP4 SHA-256 | `cb6017328d9881699a1f3b1ad650bd825fd35235c202bc0966d3d9eeb21475d6` |

The entire MP4 decoded successfully. It contains H.264 video and 32-kHz
stereo AAC, with no detected black interval ≥0.5 seconds, freeze ≥1 second, or
silence ≥0.25 seconds below −50 dB. A 24-frame manual review found an evolving
dojo board-break/acrobatics sequence followed by rooftop and courtyard action,
with no loop or collapse. Some upright and flip poses show minor stylized body
elongation.

[`run/`](run/) retains the submitted request, asynchronous job record, client
clock, normalized summary, raw `ffprobe` output, detector logs, and SHA-256.
The original MP4, animated preview, and contact sheet are in
[`../../samples/`](../../samples/README.md).
