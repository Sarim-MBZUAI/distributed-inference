"""Single-GPU greedy run of the full model, to check pp_infer.py output against."""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen3.5-4B"
prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain pipeline parallelism in two sentences."
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, ).cuda()
text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
