from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model
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

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset = load_dataset("json", data_files="./Personal-QA-DATASET.json")
    dataset = dataset["train"]
    dataset = dataset.shuffle(seed=42)

    eval_dataset = dataset.select(range(15))
    train_dataset = dataset.select(range(15, len(dataset)))

    tokenized_train = train_dataset.map(tokenize_function, batched=False)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=False)

    tokenized_train.set_format(type="torch")
    tokenized_eval.set_format(type="torch")

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                              inference_mode = False, 
                              r=8,
                              lora_alpha=32,
                              lora_dropout=0.1)
    
    model = get_peft_model(model, peft_config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        save_strategy="epoch",
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        report_to="none",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

