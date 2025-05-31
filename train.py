#Concatenate json file and apply masking to separate questions/answers (hugging face datasets)
#Auto tokenizer from hugging face
#Use peft library with transformers and accelerate with lora to finetune (loraConfig, getpeftmodel)
#Train with hugging face trainer

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def tokenize_function(example):
    prompt = f"<|im_start|>user\n{example['question']}\n<|im_end|>\n\n<|im_start|>Dylan Todd\n{example['answer']}\n<|im_end|>"

    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_attention_mask=True,
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

############################################################################################################################################################

if __name__ == "__main__":
    model_id = "rasyosef/phi-2-instruct-v0.1"

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset = load_dataset("json", data_files="./Personal-QA-DATASET.json")

    tokenized_dataset = dataset.map(tokenize_function, batched=False)
    tokenized_dataset.set_format(type="torch")
