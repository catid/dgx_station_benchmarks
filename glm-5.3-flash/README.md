# GLM-5.3-Flash

This section tracks the native-FP8
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
checkpoint at revision `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`.
It is a 320B-total, 18B-active mixture-of-experts model with 45 layers and 288
experts, 8 of which are selected per token.

## 2× DGX Station GB300

Decode uses an exact 8,192-token prompt and 1,024-token output. Rates are
aggregate output tokens per second from one distributed model server.

| C | TP2/MTP0 | TP2/MTP5 | TEP2/MTP5 |
| ---: | ---: | ---: | ---: |
| 1 | 119.0 | 181.2 | 174.1 |
| 2 | 193.7 | 272.4 | 267.4 |
| 4 | 340.8 | 285.9 | 292.7 |
| 8 | 532.2 | 490.9 | 485.0 |
| 16 | 792.7 | 728.1 | 718.4 |
| 32 | 1,001.4 | 519.5 | 516.4 |
| 64 | 1,582.1 | 1,021.6 | 1,038.2 |
| 128 | 2,019.5 | 1,569.3 | 1,549.3 |

Cold-prefill rates are prompt tokens per second.

| Context | TP2/MTP0 | TP2/MTP5 | TEP2/MTP5 |
| ---: | ---: | ---: | ---: |
| 8K | 12,645 | 14,871 | 14,214 |
| 64K | 3,721 | 15,438 | 15,076 |
| 128K | 6,519 | 15,431 | 14,873 |

## 4× RTX PRO 6000 comparison

| Decode C | MTP5 output tok/s |
| ---: | ---: |
| C1 | 148.5 |
| C2 | 219.8 |
| C4 | 324.8 |
| C8 | 458.3 |
| C10 | 522.4 |

![GLM-5.3-Flash decode throughput](charts/decode-throughput.png)

| Prefill context | Prompt tok/s |
| ---: | ---: |
| 8K | 9,936 |
| 64K | 10,152 |
| 128K | 9,892 |

![GLM-5.3-Flash prefill throughput](charts/prefill-throughput.png)

## Next lane: NVFP4

[`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/tree/aa28e1f54130286c95fee10d0705c74ce8743734)
at revision `aa28e1f54130286c95fee10d0705c74ce8743734` is a third-party
weight-only checkpoint: routed experts use NVFP4, the remainder stays BF16,
and the BF16 MTP layer is retained. This revision includes the configuration
update needed for SGLang loading. No throughput has been measured here yet.

See the [reproduction recipe](recipes/) for workload details, runtime pins,
provenance, observed concurrency, and operational notes. Machine-readable
results are in [`data/`](data/).

Return to the [repository overview](../).
