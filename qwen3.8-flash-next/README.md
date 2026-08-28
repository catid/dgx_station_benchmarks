# Qwen3.8-Flash-Next

Qwen3.8-Flash-Next NVFP4 on two DGX Station GB300 systems. This is the
Flash-Next MoE model, not Qwen3.8-27B.

## DGX Station results

The DGX headline uses
[`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89)
on SGLang. This is one distributed engine across both Stations using TP2.

| DGX profile | C1 decode | C16 decode | C64 decode | 64K cold prefill |
| --- | ---: | ---: | ---: | ---: |
| TP2/AR | 142.4 tok/s | 1,473.0 tok/s | **3,055.9 tok/s** | **24,437 tok/s** |

![DGX Station and 4× RTX PRO 6000 NVFP4 decode throughput](charts/dgx-nvfp4-decode.png)

![DGX Station and 4× RTX PRO 6000 NVFP4 cold-prefill throughput](charts/dgx-nvfp4-prefill.png)

## RTX PRO 6000 comparison

The dashed orange line is the best measured NVFP4 result from one server with
four RTX PRO 6000 Blackwell GPUs. It is a comparison point, not a DGX Station
headline.

Exact checkpoint revisions, launch settings, workload, and per-cell values are
in the [recipe](recipes/) and [`data/`](data/).

Return to the [repository overview](../).
