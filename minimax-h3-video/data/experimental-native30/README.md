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
15-second peak-HBM gate. Its request, timing, job record, media audit, and
normalized metrics are retained here if the attempt completed.
