---
language:
- zh
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- educational
---

# Chinese extraction: boundary-data diagnostic v2 (2026-10-23)

This is an **experimental, non-default adapter** on the `boundary-v2-20261023` branch. The previous smoke-test checkpoint remains on `main`. Neither is a production model.

Base: `Qwen/Qwen3-0.6B`, pinned revision `c1899de289a04d12100db370d81485cdf75e47ca`; base Apache-2.0 license included. The adapter requires the base weights.

## Fixed experiment

Both arms started from the same pinned base and matching initial adapter tensors. Each used six AdamW steps, batch 1, learning rate 0.0002, weight decay 0, gradient clipping 1, BF16 base, seed 42, rank 8 / alpha 16 / dropout 0 LoRA on q_proj and v_proj. No packing or truncation. JSON completion and end-of-turn tokens are supervised; the non-thinking prompt prefix is masked.

The control visited three original fictional notices twice. The candidate uploaded here visited those same three notices once, then three new time-first/welcome-prefix boundary notices once. The candidate is **not continued training from yesterday's adapter**.

The budget matches optimizer steps, not tokens: the control supervised 146 target tokens, candidate 136. Coverage, sample exposure and ordered recency change together; this comparison does not uniquely isolate a causal mechanism.

## Observed diagnostic results

| Subset | Base exact | Control exact | Candidate exact |
| --- | --- | --- | --- |
| 3 known regression cases | 2/3 | 1/3 | 3/3 |
| 6 added diagnostic cases | 3/6 | 2/6 | 6/6 |
| Total | 5/9 | 3/9 | 9/9 |

These are assistant-authored, visible synthetic cases, deliberately informed by previous failures. Source notices and names differ between train and dev, but template layouts overlap. **9/9 is not held-out accuracy or evidence of broad generalization.** No sealed real-world test, retention suite, multiple-seed assessment or exact training resume has been completed. The candidate is not promoted to main.

Local validation found unchanged base tensor hashes, updated adapter tensors and identical generated tokens after saving/reloading. Publication/download verification is recorded separately by the local workflow.

## Loading this branch's artifact

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

repo = "YueranCao2001/qwen3-0.6b-zh-extraction-lora"
adapter_revision = "<verified commit SHA of boundary-v2-20261023>"
tokenizer = AutoTokenizer.from_pretrained(repo, revision=adapter_revision)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca",
    torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
model = PeftModel.from_pretrained(base, repo, revision=adapter_revision).eval()
```

Use the instruction/template identified in `manifest.json`: non-thinking mode, exact keys `event`, `time`, `location`, literal input spans and JSON `null` for missing facts. Decode greedily with at most 128 new tokens. `evaluation.json` contains raw diagnostic generations. Training data, tokens/credentials, optimizer state and base weights are not uploaded. Authenticate locally to access this private repo; never put tokens in source code.
