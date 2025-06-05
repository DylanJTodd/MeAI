## GENERATES RAG DOCS FROM RAG-CORPUS

from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("rasyosef/phi-2-instruct-v0.1")

def split_into_chunks(text, max_tokens=50):
    words = text.split()
    chunks, current_chunk = [], []

    for word in words:
        current_chunk.append(word)
        if len(tokenizer(" ".join(current_chunk))["input_ids"]) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

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
