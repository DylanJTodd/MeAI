#Concatenate json file and apply masking to separate questions/answers (hugging face datasets)
#Auto tokenizer from hugging face
#Use peft library with transformers and accelerate with lora to finetune (loraConfig, getpeftmodel)
#Train with hugging face trainer

from datasets import load_dataset

dataset = load_dataset("json", data_files="./Personal-QA-DATASET.json")