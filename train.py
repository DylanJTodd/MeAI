from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model
import torch

# STEP 1: Define the EXACT system prompt you will use during inference.
# This ensures consistency between training and testing.
SYSTEM_PROMPT = """<|im_start|>system
You are an expert AI assistant role-playing as Dylan Todd. Your sole purpose is to answer questions about Dylan's professional background, skills, and projects.

**Your instructions are absolute:**
1.  You MUST answer the user's question from Dylan Todd's perspective.
2.  Your answers must be concise, professional, and direct.
3.  End every single response with '<|im_end|>'.

Failure to answer the question is not an option. Begin your response immediately as Dylan Todd.
<|im_end|>"""


def tokenize_function(example):
    # STEP 2: Construct the full prompt, exactly as it will appear during inference.
    # We use a placeholder for the RAG context. The model will learn that this
    # section exists but to focus on the question.
    prompt = f"""{SYSTEM_PROMPT}
<|im_start|>user
Here is some context to help inform your answer, note that not all of it may be relevant to the question, but it is provided to help you answer:
[Context from personal documents, resume, or projects will be provided here.]

Now answer this question directed to Dylan Todd: 
{example['question']}
<|im_end|>
<|im_start|>Dylan Todd
{example['answer']}<|im_end|>"""

    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_attention_mask=True,
    )

    input_ids = tokens["input_ids"]
    labels = input_ids.copy()

    answer_start_str = f"<|im_start|>Dylan Todd\n"
    answer_start_ids = tokenizer.encode(answer_start_str, add_special_tokens=False)

    answer_start_index = -1
    for i in range(len(input_ids) - len(answer_start_ids) + 1):
        if input_ids[i : i + len(answer_start_ids)] == answer_start_ids:
            answer_start_index = i + len(answer_start_ids)
            break
    
    if answer_start_index == -1:
        # Don't learn from this truncated example
        labels[:] = [-100] * len(labels)
    else:
        for i in range(answer_start_index):
            labels[i] = -100

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        for i in range(len(labels)):
            if input_ids[i] == pad_token_id:
                labels[i] = -100

    tokens["labels"] = labels
    return tokens

############################################################################################################################################################

if __name__ == "__main__":
    model_id = "rasyosef/phi-2-instruct-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
    model.gradient_checkpointing_enable()

    model.resize_token_embeddings(len(tokenizer))


    dataset = load_dataset("json", data_files="./Personal-QA-DATASET.json")
    dataset = dataset["train"]
    dataset = dataset.shuffle(seed=42)

    eval_dataset = dataset.select(range(15))
    train_dataset = dataset.select(range(15, len(dataset)))

    tokenized_train = train_dataset.map(tokenize_function, batched=False, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=False, remove_columns=eval_dataset.column_names)

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
        num_train_epochs=3,
        save_strategy="epoch",
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
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