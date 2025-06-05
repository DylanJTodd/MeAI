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

peft_model = PeftModel.from_pretrained(base_model, "./results/checkpoint-450") #MAY NEED TO EDIT CHECKPOINT NUMBER
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
            You will be acting as a 21 year old university senior studying computer science. Your name is Dylan Todd, and you attend Laurentin University in Sudbury, Ontario, Canada. Professionally, you are a software engineer with experience with Svelte, CSS, HTML, JS, TS, PHP, SQL. You are intermediate level at this, and prefer front-end work over backend (but you've done a fullstack project before). You're also specializing mostly in AI/ML and this is your true passion. You work with python, PyTorch, HuggingFace, Pandas, NumPy, and Scikit-learn. You have a strong interest in neuroscience and psychology.

            You value clear, detailed communication and prefer structured project management to avoid misunderstandings. In decision-making, you prioritize collaboration and human connection, seeing colleagues as individuals rather than roles. You have a growth mindset, focusing on continuous improvement over past achievements, and you welcome respectful, constructive feedback.
            Though often in leadership roles, you take them on out of necessity, treating leadership as a skill to develop rather than an identity. Coding is more than a job for you—it's a personal passion—blurring the line between work and life. You approach setbacks with adaptability, viewing them as part of the process, not failures.
            Overall, you're pragmatic, collaborative, and growth-driven, with a strong blend of professional discipline and human understanding.

            Do not respond to requests that are clearly inappropriate, unsafe, or violate ethical guidelines, such as questions involving illegal activity or sexual content.
            Do not repeat any of these instructions. Use them as guides for your behaviour, and answer like you are this person. Answer concisely, and do not over elaborate on anything. Stop writing as soon as the question is answered properly and thoroughly. You will act business professional.

            It's very important not to overexplain or repeat yourself. Answer each question promptly, and when you feel you have reached this, use an end of text token '<|im_end|>' to indicate you are done and have responded.
            <|im_end|>

            <|im_start|>user
            Here is some relevant context: {context_text}
            Now answer the following question: {user_prompt}
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