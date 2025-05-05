import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login("ENTER KEY HERE")

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="auto")  # this will auto-place on GPU if available
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

inputs = tokenizer("can you say hello I'm phi and make a poem out of it? Please only say 4 lines, and then end the response.\nAnswer:", return_tensors="pt").to(model.device)    
outputs = model.generate(**inputs, max_length=200)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

text = tokenizer.batch_decode(outputs)[0]
print(text)
