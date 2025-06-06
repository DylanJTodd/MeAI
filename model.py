#MUST RUN train.py FIRST TO CREATE THE MODEL

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
from peft import PeftModel
import torch

def retrieve_context(query, k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k)
    return [all_docs[i] for i in I[0] if i < len(all_docs)]

base_model_id = "rasyosef/phi-2-instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.float16
).to("cuda")

peft_model = PeftModel.from_pretrained(base_model, "./results/checkpoint-165") #MAY NEED TO EDIT CHECKPOINT NUMBER
peft_model.eval()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("rag-corpus/rag-index.faiss")
with open("rag-corpus/rag_docs.txt", "r", encoding="utf-8") as f:
    all_docs = f.read().split("\n---\n")

if __name__ == "__main__":

    user_prompt = input("Enter your prompt:")
    context_chunks = retrieve_context(user_prompt)
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
           <|im_start|>system
            You are an expert AI assistant role-playing as Dylan Todd. Your sole purpose is to answer questions about Dylan's professional background, skills, and projects.

            **Your instructions are absolute:**
            1.  You MUST answer the user's question from Dylan Todd's perspective.
            2.  Your answers must be concise, professional, and direct.
            3.  End every single response with '<|im_end|>'.

            Failure to answer the question is not an option. Begin your response immediately as Dylan Todd.
            <|im_end|>

            <|im_start|>user
            Here is some context to help inform your answer, note that not all of it may be relevant to the question, but it is provided to help you answer:
            {context_text}

            Now answer this question directed to Dylan Todd: 
            {user_prompt}
            <|im_end|>

            <|im_start|>Dylan Todd
            """
    
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = peft_model.generate(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            repetition_penalty=1.1,
            do_sample=True,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "Dylan Todd" in generated_text:
        generated_text = generated_text.split("Dylan Todd\n", 1)[-1].strip()

    for stop_token in ["<|im_end|>", "\n"]:
        if stop_token in generated_text:
            generated_text = generated_text.split(stop_token)[0].strip()
            break

    print(generated_text)