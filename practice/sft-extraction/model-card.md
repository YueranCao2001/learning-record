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

# Qwen3 0.6B Chinese extraction: six-step LoRA smoke test

This is a diagnostic adapter, **not a production model or a demonstrated capability improvement**. It requires the original Qwen base model. The original base license applies; see LICENSE.

## Training scope

- Base: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Three fictional, assistant-authored Chinese event notices used for training; three visible synthetic dev notices used only for generation checks. No sealed test set.
- Task: extract literal spans into JSON keys `event`, `time`, `location`; missing values are `null`.
- Six optimizer steps, batch size 1; no packing or truncation. The three training notices are visited twice in a fixed order.
- LoRA rank 8, alpha 16, dropout 0, q_proj/v_proj only; AdamW learning rate 0.0002, weight decay 0, gradient clipping 1.0, seed 42.
- BF16 base, assistant-completion-only next-token loss (JSON plus end-of-turn token); the non-thinking prefix is context, not a supervised target.

The uploaded manifest fixes data/template/code hashes and generation settings. `evaluation.json` contains raw diagnostic outputs and scores. Do not interpret performance on these three visible cases as general accuracy. Saving and inference reload do not demonstrate exact training resume.

## Observed result (2026-10-22)

The fixed visible dev cases went from 2/3 exact matches before training to 1/3 after six steps. All outputs were parseable and schema-valid, but event-name extraction regressed. Lower training loss did not establish better generation quality. This adapter is retained as a reproducible pipeline diagnostic, not promoted over the base model.

The frozen base tensor hash was unchanged, 112 adapter tensors changed, and saved/reloaded adapter tensors and greedy generation token sequences matched exactly locally. Remote download/load verification is recorded separately by the publishing workflow. The `uploaded` flag inside the local training summary describes its pre-publication stage, not the current Hub repository state.

## Loading

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

repo = "YueranCao2001/qwen3-0.6b-zh-extraction-lora"
# Pin this adapter's commit in real usage; the upload report records the tested commit.
adapter_revision = "<adapter commit SHA>"
base_revision = "c1899de289a04d12100db370d81485cdf75e47ca"
tokenizer = AutoTokenizer.from_pretrained(repo, revision=adapter_revision)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", revision=base_revision,
    torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
model = PeftModel.from_pretrained(base, repo, revision=adapter_revision).eval()
messages = [
    {"role": "system", "content": "提取活动名称、时间和地点。只输出 JSON 对象，键为 event、time、location。值必须逐字来自输入；未提供的字段用 null。不要推断日期或补全地点。"},
    {"role": "user", "content": "周六15:00举行手工沙龙，举办地点待通知。"},
]
ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    enable_thinking=False, return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = model.generate(
        ids, attention_mask=torch.ones_like(ids), do_sample=False,
        max_new_tokens=128, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True))
```

Use local Hugging Face authentication to access this private repository. Do not put credentials into code. Invalid JSON, unsupported facts, and regressions remain possible. Training notices, credentials, base weights, optimizer state and local caches are not part of this upload.
