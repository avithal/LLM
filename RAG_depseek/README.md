# 📄 Retrieval-Augmented Generation (RAG) on PDF Documents

This project demonstrates a **simple, end-to-end RAG pipeline** using **LangChain**, **Ollama**, and **FAISS** to ask questions over a PDF document.

The pipeline:

1. Loads a PDF
2. Splits it into chunks
3. Embeds the chunks
4. Stores them in a vector database (FAISS)
5. Retrieves relevant chunks for a question
6. Uses an LLM to answer **only based on the retrieved context**

---

## 🧠 Model

* **LLM**: `deepseek-r1:8b` (via Ollama)
* **Embeddings**: Ollama embeddings using the same model

> You can easily swap to another Ollama-supported model.

---

## 📁 Project Structure

```
.
├── duh.pdf                 # Input PDF document
├── rag_pdf.py              # Main RAG pipeline script
├── README.md               # Project documentation
```

---

## ⚙️ Requirements

### Python

* Python 3.9+

### Install Dependencies

```bash
pip install langchain langchain-community langchain-text-splitters \
            langchain-ollama faiss-cpu pypdf
```

### Ollama

Install Ollama and pull the model:

```bash
ollama pull deepseek-r1:8b
```

Make sure Ollama is running:

```bash
ollama serve
```

---

## 🧩 Pipeline Overview

### 1️⃣ Load PDF

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("duh.pdf")
pages = loader.load()
```

---

### 2️⃣ Split into Chunks

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100
)

chunks = splitter.split_documents(pages)
```

---

### 3️⃣ Create Embeddings & Vector Store

```python
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model=MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()
```

---

### 4️⃣ Load LLM

```python
from langchain_ollama import ChatOllama

model = ChatOllama(model=MODEL, temperature=0)
```

---

### 5️⃣ Prompt Template

The model is **forced to answer only from context**:

```python
from langchain.prompts import PromptTemplate

template = """
You are an assistant that provides answers to questions based on
provided context.

If the answer is not in the context, reply: "I don't know".

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate.from_template(template)
```

---

### 6️⃣ Full RAG Chain

```python
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | StrOutputParser()
)
```

---

## ❓ Asking Questions

```python
questions = [
    "What can you get away with when you only have a small number of users?",
    "What's the most common unscalable thing founders have to do at the start?",
]

for q in questions:
    print(chain.invoke({"question": q}))
```

---

## ✅ Key Features

* 📄 PDF-based knowledge grounding
* 🔍 Semantic retrieval with FAISS
* 🧠 LLM answers constrained to context (low hallucination)
* 🔌 Runs **fully locally** (no cloud APIs)

---

## 🚀 Possible Extensions

* Add **metadata filtering** (page number, section)
* Persist FAISS index to disk
* Use **multi-PDF ingestion**
* Add **citations / source highlighting**
* Wrap with **Streamlit or Gradio UI**

---

## 🛡️ Hallucination Control

This project reduces hallucinations by:

* Strict context-based prompting
* Temperature = 0
* Retrieval before generation (RAG)

---

## 📌 Notes

* Designed for experimentation and learning
* Suitable for research papers, startup docs, or technical PDFs

---

## 🧑‍💻 Author

Avithal — Computer Vision & AI Engineer

---

⭐ If this helped you, consider starring the repo!
