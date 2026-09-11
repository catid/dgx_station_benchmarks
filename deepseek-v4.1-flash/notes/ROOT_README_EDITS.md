# Proposed edits to the repository root README and test registry

These edits are **not applied** by the section scaffold; apply them by hand once
the accepted rows exist. Placeholders follow the section README.

## 1. Experiments table row

Insert directly after the `[DeepSeek-V4-Flash-0731]` row (keeps the DeepSeek
rows adjacent). Three cells, `tok/s` units, `;`-separated facts, no trailing
period:

```markdown
| [DeepSeek-V4.1-Flash](deepseek-v4.1-flash/) | Official native FP8-dense/FP4-expert checkpoint; 2× SGLang TP2+EP2 and vLLM PP2/TP2 over Data Direct RDMA (1× not attempted) | Prefill with SWA bounded replay: 16K C1 39,549 prompt tok/s; 128K C16 37,693 aggregate prompt tok/s (exact full prefill 26,316 and 24,419); DSpark C1 180.0 output tok/s |
```

## 2. Contents block

Insert after the `deepseek-v4-flash-0731/` block, same box-drawing style
(`tests/` is deliberately not listed, matching the other sections):

```text
├── deepseek-v4.1-flash/
│   ├── README.md
│   ├── charts/
│   ├── data/
│   ├── notes/
│   └── recipes/
```

## 3. Shared methodology

1. Disambiguate the existing sentence so "DeepSeek" no longer implies the new
   section:

   `Qwen3.8-27B, DeepSeek, and the sealed external Qwen3.8-Flash-Next matrix use its finite-request layer:`
   →
   `Qwen3.8-27B, DeepSeek-V4-Flash-0731, DeepSeek-V4.1-Flash decode, and the sealed external Qwen3.8-Flash-Next matrix use its finite-request layer:`

2. Append after the sustained-decode paragraph:

```markdown
DeepSeek-V4.1-Flash prefill does not use `llm-inference-bench`. Its prefill
tables come from the section's own `bench_prefill.py` client against each
engine's native endpoint: unique random-token prompts of exactly 16K, 32K,
64K, and 128K tokens, one generated token, temperature 0, a fixed 1, 4, or 16
requests held in flight, a cache flush before every point, and aggregate
prompt tokens divided by the wave's wall time. Those aggregate multi-request
prefill values are not interchangeable with the single-request cold-prefill
cells reported by the `llm-inference-bench` sections. Its decode rows use the
finite-request layer above with the same pinned client.
```

## 4. Offline publication checks sentence

`The repository-level suite currently verifies the GLM-5.3-Flash and Qwen3.8-Flash-Next imported rows, …`
→
`The repository-level suite currently verifies the GLM-5.3-Flash, Qwen3.8-Flash-Next, and DeepSeek-V4.1-Flash rows, …`

## 5. Footer date

`Measured August 2026.` → `Measured August–September 2026.`

## 6. `tests/test_publication_sections.py`

Append to `SECTION_TESTS`:

```python
    (
        "deepseek_v4_1_flash_section_contract",
        REPOSITORY / "deepseek-v4.1-flash/tests/test_section_contract.py",
    ),
```

and change the `load_tests` docstring from
`"""Load the two section suites directly, without recursive discovery."""`
to `"""Load each section suite directly, without recursive discovery."""`.

Verify with `python3 -m unittest discover -v` from the repository root.
