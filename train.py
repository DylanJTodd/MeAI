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

    input_ids = tokens["input_ids"]
    labels = input_ids.copy()


    # Label mask for the answer rather than the entire input
    answer_start_str = f"<|im_start|>Dylan Todd\n"
    answer_start_ids = tokenizer.encode(answer_start_str, add_special_tokens=False)

    for index in range(len(input_ids) - len(answer_start_ids) + 1):
        if input_ids[index : index + len(answer_start_ids)] == answer_start_ids:
            answer_start_index = index + len(answer_start_ids)
            break
    else:
        answer_start_index = len(input_ids)

    IGNORE_INDEX = -100
    for index in range(answer_start_index):
        labels[index] = IGNORE_INDEX

    tokens["labels"] = labels
    return tokens

############################################################################################################################################################

if __name__ == "__main__":
    model_id = "rasyosef/phi-2-instruct-v0.1"

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset = load_dataset("json", data_files="./Personal-QA-DATASET.json")

    tokenized_dataset = dataset.map(tokenize_function, batched=False)
    tokenized_dataset.set_format(type="torch")
