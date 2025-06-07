# MeAI: Ask Dylan Todd Anything

**MeAI** is a personalized RAG-powered AI chatbot designed to answer questions about my background, skills, and projects. It combines a fine-tuned [Phi-2 Instruct model](https://huggingface.co/rasyosef/phi-2-instruct-v0.1) with sentence embeddings and document retrieval for accurate, context-aware answers.


## 🙋 About

This project is built by **Dylan Todd** to demonstrate a personalized, self-representing AI agent. It combines fine-tuning, retrieval-augmented generation(RAG), and prompt specialization to simulate informed, professional conversations.

---

## 🚀 Try It on Gradio (Slow but Free)

You can try the app live via Gradio here:  
🔗 [huggingface.co/spaces/DylanJTodd/MeAI](https://huggingface.co/spaces/DylanJTodd/MeAI)

> ⚠️ **Note**: Response times may take **up to 10 minutes** on the free CPU tier. For faster performance, run locally with a GPU.

---

## 🧠 How It Works

This chatbot uses:
- `SentenceTransformer` to embed and retrieve top documents
- `FAISS` for fast similarity search
- A fine-tuned PEFT LoRA model on top of Phi-2
- Custom prompt engineering to act as "Dylan Todd"

---

## 💻 Run It Locally (Recommended)

You’ll get much better performance if you run MeAI locally on a GPU machine.

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/meai.git
cd meai
````

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run Setup Scripts

You'll need to:

1. Fine-tune the model
2. Preprocess the documents
3. Build the FAISS index
4. Launch the app

You can run each step individually:

```bash
python train.py
python preprocess_chunks.py
python rag-index.py
python model.py
```
---

## 🧩 Tech Stack

* [Transformers](https://huggingface.co/docs/transformers/index)
* [Datasets](https://huggingface.co/docs/datasets/index)
* [PEFT](https://github.com/huggingface/peft)
* [Gradio](https://gradio.app)
* [FAISS](https://github.com/facebookresearch/faiss)
* [SentenceTransformers](https://www.sbert.net/)

---
