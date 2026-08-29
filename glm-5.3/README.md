# GLM-5.3 NVFP4 + DFlash2

Full-size
[`incoai/GLM-5.3-NVFP4@54e5252`](https://huggingface.co/incoai/GLM-5.3-NVFP4/tree/54e52520606f96b3d9fc84088ad22882a61648ac)
with
[`incoai/GLM-5.3-DFlash2@425aa61`](https://huggingface.co/incoai/GLM-5.3-DFlash2/tree/425aa615ce320caac34400208b30808c8f14f76c)
on 2× NVIDIA DGX Station GB300.

## Headline results

| Goal | Winning profile | Result | Residency / latency |
| --- | --- | ---: | --- |
| C1 code | SGLang TP2+EP2 · DFlash2 K7 | **165.5 tok/s** | 5.115 ms median ITL |
| C1 prose | SGLang TP2+EP2 · DFlash2 K7 | **107.4 tok/s** | 8.206 ms median ITL |
| C2 code | vLLM PP2 42/36 · DFlash2 K5 | **270.6 tok/s** | 1.9 / 2 active · 0.0% queued |
| C4 code | vLLM PP2 42/36 · DFlash2 K5 | **409.9 tok/s** | 3.8 / 4 active · 0.0% queued |
| C8 code | vLLM PP2 42/36 · DFlash2 K7 | **553.8 tok/s** | 7.6 / 8 active · 2.7% queued |
| C16 code | vLLM PP2 42/36 · DFlash2 K7 | **726.8 tok/s** | 14.9 / 16 active · 3.6% queued |
| C32 code | vLLM PP2 42/36 · DFlash2 K7 | **1,065.0 tok/s** | 29.9 / 32 active · 9.2% queued |
| C64 code | vLLM PP2 42/36 · DFlash2 K7 | **1,093.8 tok/s** | 60.2 / 64 active · 11.1% queued |
| Exact 64K cold prefill | SGLang PP2/AR 40/38 | **25,893 prompt tok/s** | 2.531 s median TTFT |

![GLM-5.3 output tokens per second per user](charts/per-user-throughput.png)

![GLM-5.3 decode throughput](charts/decode-throughput.png)

## PP2 DFlash2 proposal sweep

| Proposals | C1 code | C1 prose | C2 | C4 | C8 | C16 | C32 | C64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **K4** | **154.9** | **106.3** | 270.6 | 399.9 | 542.5 | 725.6* | — | — |
| **K5** | 152.6 | 104.1 | **270.6** | **409.9** | 538.9 | 723.6 | — | — |
| **K7** | 146.0 | 92.7 | 267.0 | 405.7 | **553.8** | **726.8** | **1,065.0** | **1,093.8** |

Aggregate output tok/s, exact 8,192-token input and 1,024-token output. All
cells use `5×C` measured requests except the validated `K4 C16` `2×C` screen.

![GLM-5.3 exact 64K cold prefill](charts/prefill-throughput.png)

## 4× RTX PRO 6000 comparison — different model stack

**EXL3 3.25 bpw derivative checkpoint · native MTP3**

| Decode | Mode | Aggregate output tok/s | Max resident |
| --- | --- | ---: | ---: |
| C1 code | MTP3 | **60.03** | 1 |
| C1 prose | AR | **41.53** | 1 |
| C8 code | MTP3 | **113.63** | 8 |
| C16 code | MTP3 | **158.85** | 16 |
| C32 code | MTP3 | **164.64** | 29 |

| Cold prefill | Prompt tok/s | TTFT |
| ---: | ---: | ---: |
| 8K | **2,937.94** | 2.788 s |
| 64K | **2,698.40** | 24.290 s |
| 128K | **2,473.41** | 52.999 s |

![4× RTX PRO 6000 Blackwell comparison](charts/workstation-comparison.png)

This is a comparison point, not the exact Inco NVFP4 target or DFlash2 draft
used in the DGX Station tables.

[`PP2 + DFlash2 recipe`](recipes/pp2_dflash2.md) ·
[`General recipe`](recipes/) · [`Data`](data/)

Return to the [repository overview](../).
