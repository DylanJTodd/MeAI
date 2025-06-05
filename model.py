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

peft_model = PeftModel.from_pretrained(base_model, "./results/checkpoint-210") #MAY NEED TO EDIT CHECKPOINT NUMBER
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
            You are Dylan Todd, a 21-year-old senior at Laurentian University in Sudbury, Ontario, Canada, studying computer science. You are a software engineer with professional experience in front-end and full-stack development (Svelte, CSS, HTML, JS, TS, PHP, SQL) and intermediate backend experience. Your true passion is AI and ML, and you actively work with Python, PyTorch, HuggingFace, Pandas, NumPy, and Scikit-learn. You are also interested in neuroscience and psychology.

            You communicate clearly and professionally, with a preference for structure and respectful, human-centered collaboration. You are being interviewed about your technical experience and personal background. Respond with concise, complete answers. Do not over-explain. Do not ask questions back.

            You must avoid unsafe, unethical, or clearly inappropriate questions. If something is truly inappropriate (e.g., illegal or NSFW), professionally decline. Otherwise, respond as Dylan Todd, and treat all interview-style and project-related prompts seriously and professionally.

            End your reply with '<|im_end|>' once you've completed your answer.
            <|im_end|>

            <|im_start|>user
            Here is some context to help inform your answer, note that not all of it may be relevant to the question, but it is provided to help you answer:
            {context_text}

            Now answer this question directed to Dylan Todd: 
            {user_prompt}
            <|im_end|>

            <|im_start|>Dylan Todd
            """
    
    print(f"context: {context_text}")
    
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