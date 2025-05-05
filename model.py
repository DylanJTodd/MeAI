from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

model_id = "rasyosef/phi-2-instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = """<|im_start|>system
You are a helpful assistant.
<|im_end|>

<|im_start|>user
Can you say hello I'm phi and make a poem out of it? Please only say 4 lines, and then end the response.<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generation_config = GenerationConfig(
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

output = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(output[0], skip_special_tokens=True))
