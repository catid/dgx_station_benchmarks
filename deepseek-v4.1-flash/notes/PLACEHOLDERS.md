# Placeholder ledger

Every `{{TOKEN}}` still present in this section's Markdown files, so nothing is missed when the rows land.
Fill a token only from `data/*.csv` (never by hand from a log), keep the formatting rule shown, and delete the row here
once it is gone from every file. `tests/test_section_contract.py` refuses a placeholder that stands where its lane
already has rows (accepted or diagnostic), a replaced number that does not round from the CSV, and a table (the README best-numbers table or a notes/ detailed table) that
omits a configuration the charts draw. A `—` cell means the point is outside that lane's measured grid, never a
placeholder for a pending run.

| Token | Files | Source and format |
| --- | --- | --- |
| `{{STATUS_VLLM_TP2_DECODE}}` | notes/README.md | lane disposition wording for the 'What was measured' table (accepted / pending / diagnostic only / failed: reason) |
| `{{STATUS_VLLM_TP2_DSPARK_DECODE}}` | notes/README.md | lane disposition wording for the 'What was measured' table (accepted / pending / diagnostic only / failed: reason) |
| `{{VLLM_TP2_DECODE_C16}}` | notes/README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_DECODE_C1}}` | notes/README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_DECODE_C32}}` | notes/README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_DECODE_C4}}` | notes/README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_DECODE_C64}}` | README.md, notes/README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_DECODE_C16}}` | notes/README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_DECODE_C1}}` | notes/README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_DECODE_C32}}` | notes/README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_DECODE_C4}}` | notes/README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_DECODE_C64}}` | README.md, notes/README.md | throughput.csv aggregate_output_tokens_per_second, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_USER_C16}}` | notes/README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_USER_C1}}` | README.md, notes/README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_USER_C32}}` | notes/README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_USER_C4}}` | notes/README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_DSPARK_USER_C64}}` | notes/README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 DSpark, f'{v:,.1f}' |
| `{{VLLM_TP2_USER_C16}}` | notes/README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_USER_C1}}` | README.md, notes/README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_USER_C32}}` | notes/README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_USER_C4}}` | notes/README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 AR, f'{v:,.1f}' |
| `{{VLLM_TP2_USER_C64}}` | notes/README.md | throughput.csv per_user_output_tokens_per_second_p50, vLLM TP2 AR, f'{v:,.1f}' |

22 placeholders.
