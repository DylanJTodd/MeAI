## GENERATES RAG DOCS FROM RAG-CORPUS

from transformers import AutoTokenizer
import os
import re

tokenizer = AutoTokenizer.from_pretrained("rasyosef/phi-2-instruct-v0.1")

def split_into_chunks(text, max_tokens=80):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        tentative = current_chunk + " " + sentence if current_chunk else sentence
        tokenized = tokenizer(tentative, truncation=False, return_tensors="np")
        token_count = len(tokenized["input_ids"][0])

        if token_count <= max_tokens:
            current_chunk = tentative
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def process_all_files(root_dir="rag-corpus", output_file="rag-corpus/rag_docs.txt"):
    all_chunks = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    chunks = split_into_chunks(text, max_tokens=50)
                    all_chunks.extend(chunks)

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("\n---\n".join(all_chunks))

if __name__ == "__main__":
    process_all_files()
