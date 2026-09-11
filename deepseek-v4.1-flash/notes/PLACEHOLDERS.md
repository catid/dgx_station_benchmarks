# Placeholder ledger

Every `{{TOKEN}}` still present in this section's Markdown files, so nothing is missed when the rows land.
Fill a token only from `data/*.csv` (never by hand from a log), keep the formatting rule shown, and delete the row here
once it is gone from every file. `tests/test_section_contract.py` refuses a placeholder that stands where its lane
already has rows (accepted or diagnostic), a replaced number that does not round from the CSV, and a README table that
omits a configuration the charts draw. A `—` cell means the point is outside that lane's measured grid, never a
placeholder for a pending run.

| Token | Files | Source and format |
| --- | --- | --- |
| `{{STATUS_VLLM_TP2_DECODE}}` | README.md | lane disposition wording for the 'What was measured' table (accepted / pending / diagnostic only / failed: reason) |
| `{{STATUS_VLLM_TP2_DSPARK_DECODE}}` | README.md | lane disposition wording for the 'What was measured' table (accepted / pending / diagnostic only / failed: reason) |
| `{{STATUS_VLLM_TP2_PREFILL}}` | README.md | lane disposition wording for the 'What was measured' table (accepted / pending / diagnostic only / failed: reason) |
| `{{VLLM_IMAGE_ID}}` | recipes/README.md | docker image inspect --format '{{.Id}}' of the vLLM image on the measured nodes |
| `{{VLLM_TP2_DECODE_C16}}` | README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_DECODE_C1}}` | README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_DECODE_C32}}` | README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_DECODE_C4}}` | README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_DECODE_C64}}` | README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_DECODE_C16}}` | README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_DECODE_C1}}` | README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_DECODE_C32}}` | README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_DECODE_C4}}` | README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_USER_C16}}` | README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_USER_C1}}` | README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_USER_C32}}` | README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_USER_C4}}` | README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_PREFILL_128K_BEST}}` | README.md | prefill.csv aggregate_prompt_tokens_per_second, vllm/cross_node_tp2/ar, f'{v:,.0f}' |
| `{{VLLM_TP2_PREFILL_16K_BEST}}` | README.md | prefill.csv aggregate_prompt_tokens_per_second, vllm/cross_node_tp2/ar, f'{v:,.0f}' |
| `{{VLLM_TP2_PREFILL_32K_BEST}}` | README.md | prefill.csv aggregate_prompt_tokens_per_second, vllm/cross_node_tp2/ar, f'{v:,.0f}' |
| `{{VLLM_TP2_PREFILL_64K_BEST}}` | README.md | prefill.csv aggregate_prompt_tokens_per_second, vllm/cross_node_tp2/ar, f'{v:,.0f}' |
| `{{VLLM_TP2_PREFILL_64K_C16}}` | README.md | prefill.csv aggregate_prompt_tokens_per_second, vllm/cross_node_tp2/ar, f'{v:,.0f}' |
| `{{VLLM_TP2_PREFILL_64K_C1}}` | README.md | prefill.csv aggregate_prompt_tokens_per_second, vllm/cross_node_tp2/ar, f'{v:,.0f}' |
| `{{VLLM_TP2_PREFILL_64K_C4}}` | README.md | prefill.csv aggregate_prompt_tokens_per_second, vllm/cross_node_tp2/ar, f'{v:,.0f}' |
| `{{VLLM_TP2_TTFT_128K_C1}}` | README.md | prefill.csv ttft_p50_seconds at C1, vLLM TP2 lane, f'{v:.3f}s' |
| `{{VLLM_TP2_TTFT_16K_C1}}` | README.md | prefill.csv ttft_p50_seconds at C1, vLLM TP2 lane, f'{v:.3f}s' |
| `{{VLLM_TP2_TTFT_32K_C1}}` | README.md | prefill.csv ttft_p50_seconds at C1, vLLM TP2 lane, f'{v:.3f}s' |
| `{{VLLM_TP2_TTFT_64K_C1}}` | README.md | prefill.csv ttft_p50_seconds at C1, vLLM TP2 lane, f'{v:.3f}s' |
| `{{VLLM_TP2_USER_C16}}` | README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_USER_C1}}` | README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_USER_C32}}` | README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_USER_C4}}` | README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_USER_C64}}` | README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 AR, f'{v:,.1f}' |

33 placeholders.
