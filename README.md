AmbedkarGPT — Retrieval-Augmented Question Answering System

AmbedkarGPT is a lightweight RAG (Retrieval-Augmented Generation) project that answers questions about Dr. B.R. Ambedkar’s speech using:

LangChain 0.3+ (New LCEL Pipeline)

ChromaDB Vector Database

HuggingFace MiniLM Embeddings

Ollama (Mistral) as LLM

Python 3.12

Local-only setup (no API keys required)

Features

Latest LangChain LCEL RAG pipeline
Uses ChromaDB as vector store
HuggingFace MiniLM-L6-v2 for embeddings
Runs fully offline via Ollama + Mistral
Compatible with Python 3.12
Clean and simple CLI Q&A interface
Fast, accurate retrieval-based answering

Installation
1️ Install Python 3.12+

Download: https://www.python.org/downloads/

2️ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

3️ Install dependencies

Use this requirements.txt (compatible with Python 3.12):

langchain
langchain-community
langchain-core
langchain-text-splitters
chromadb
sentence-transformers
transformers
numpy
pydantic
typing_extensions


Install them:

pip install -r requirements.txt


Setting Up Ollama
Install Ollama

Download from: https://ollama.ai

Pull the Mistral model
ollama pull mistral

Start Ollama server
ollama serve

How It Works (RAG Pipeline)

Load text from speech.txt

Split into chunks using CharacterTextSplitter

Create vector embeddings using HuggingFace MiniLM

Store embeddings inside ChromaDB

Accept user question

Retrieve relevant text chunks

Combine context + question using LCEL

Generate an answer via Ollama (Mistral)
