# InfoStream-AI
A Streamlit-powered RAG chatbot providing real-time, sourced answers from documents using an open-source LLM and vector database.

## Setup

```bash
git clone <https://github.com/Upenderok/InfoStream-AI.git>
cd InfoStream-AI
source .venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

## Demo
Watch a short demonstration of the chatbot in action, showcasing its ability to stream responses and answer document-grounded questions:
[**Watch Chatbot Demo Video on Loom**](https://www.loom.com/share/07628c94b8ab42a1beabe103b041dda3?sid=db771688-089e-44c1-94b3-bff4e9d5916b)

## Technical Report
For a detailed technical overview of the system's architecture, including chunking logic, embedding model, prompt format, and performance notes, please refer to the accompanying PDF report:
**In the main Directory**(RAG_Report.pdf)

## How It Works (High-Level Architecture)

The chatbot employs a standard RAG pipeline:

Document Ingestion & Chunking: The input PDF document (AI Training Document.pdf) is processed using a sentence-aware splitter to break it into smaller, overlapping chunks (e.g., 120 tokens with 20% overlap).

Embedding & Vector Store: Each chunk is converted into a numerical vector (embedding) using the BAAI/bge-small-en-v1.5 embedding model. These embeddings are then stored in a FAISS IndexFlatIP vector database for efficient semantic search.

Retrieval: When a user asks a question, a semantic search is performed on the vector database to retrieve the most relevant document chunks based on the query's meaning.

Generation: The retrieved chunks, along with the user's question and a highly specific system prompt, are fed into the local Large Language Model (Phi-3-mini-4k-instruct-q4.gguf). The LLM then synthesizes a grounded answer, adhering strictly to the provided context.

Setup and Installation
Follow these steps to get the chatbot up and running on your local machine.

Prerequisites
Python 3.8+

Git (optional, but recommended for cloning the repository)

Steps
Clone the repository (if applicable):

git clone https://github.com/YourGitHubUsername/InfoStream-AI.git
cd InfoStream-AI

(Replace https://github.com/YourGitHubUsername/InfoStream-AI.git with your actual repository URL)

Create and activate a virtual environment:

python -m venv .venv
# On Windows (Git Bash/MinGW64):
source .venv/Scripts/activate
# On Windows (CMD/PowerShell):
# .\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

(Ensure your requirements.txt is up-to-date by running pip freeze > requirements.txt in your activated environment if you haven't already).

Download the LLM (Phi-3-mini GGUF):

Create a models directory in your project root: mkdir models

Download the Phi-3-mini-4k-instruct-q4.gguf file from TheBloke's Hugging Face repository (search for "TheBloke/Phi-3-mini-4k-instruct-GGUF" on Hugging Face).

Place the downloaded .gguf file into the models directory.

Place the Document:

Ensure your document, AI Training Document.pdf, is located in the root of your project directory.

Usage
Process the document (Chunking & Embedding):
First, you need to process the document to create the chunks and the vector database.

python src/chunker.py # Assuming this script handles both chunking and embedding
# OR if you have separate scripts:
# python src/chunker.py
# python src/embedder.py

(Adjust the command above based on how you trigger your chunking/embedding process)

Run the Streamlit application:

streamlit run app.py

Access the Chatbot:
Open your web browser and navigate to the Local URL provided in the terminal.

Project Structure
.
├── .venv/                   # Python Virtual Environment
├── chunks/                  # Directory for processed document chunks
├── models/                  # Directory for downloaded LLM (e.g., Phi-3-mini GGUF)
├── src/
│   ├── __init__.py
│   ├── chunker.py           # Script for document chunking
│   ├── embedder.py          # Script for generating embeddings and building vector store
│   ├── generator.py         # LLM interaction and prompt formatting
│   └── retriever.py         # Logic for retrieving relevant chunks from vector store
├── vectordb/                # Directory for FAISS index (vector database)
├── AI Training Document.pdf # The primary document for the chatbot
├── app.py                   # Main Streamlit application
├── RAG_Report.pdf           # Detailed technical report
├── README.md                # This file
└── requirements.txt         # Python dependencies

Acknowledgements

Hugging Face: For providing access to pre-trained models and a vibrant open-source community.

llama.cpp / llama-cpp-python: For enabling efficient local LLM inference on consumer hardware.

Streamlit: For simplifying the creation of interactive


License

This project is licensed under the MIT License.
(You should create a LICENSE file in your root directory if you want to explicitly define this.)