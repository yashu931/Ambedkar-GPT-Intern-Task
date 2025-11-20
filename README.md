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


ASSIGNMENT-02
RAG Evaluation Framework for QA System
Overview
This project implements a retrieval-augmented generation (RAG) evaluation framework for a question answering system over a document corpus. It quantitatively measures retrieval accuracy and answer quality using NLP metrics such as Hit Rate, MRR, Precision@K, ROUGE-L, BLEU, and cosine similarity.

The framework supports evaluating different chunk sizes of the corpus (small, medium, large) and generates detailed per-question and aggregate performance metrics.

Directory Structure
text
├── corpus/                # Folder containing text documents to build corpus
├── test_dataset.json      # JSON file of test questions with ground truth answers
├── testresults.json       # Output JSON file storing evaluation results
├── evaluation.py          # Main evaluation script to run
├── requirements.txt       # Python dependencies for the project
Setup Instructions
Set up a Python 3.8+ virtual environment:

bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install required dependencies:

bash
pip install -r requirements.txt
The main dependencies include:

sentence-transformers

rouge-score

scikit-learn

nltk

requests

Download NLTK 'punkt' tokenizer if missing:

python
import nltk
nltk.download('punkt')
Preparing Your Data
Place your corpus documents as plain text files inside the corpus/ directory.

Ensure test_dataset.json contains your test queries and associated ground truth answers in the expected format.

Running the Evaluation
Run the evaluation with:

bash
python evaluation.py
The script will:

Load the corpus and test dataset.

Create embeddings for chunks of text at specified sizes (small=250, medium=550, large=900 chars).

Retrieve top-k relevant chunks per question.

Query a local Ollama Mistral 7B LLM server (localhost:11434) for generating answers.

Calculate retrieval metrics (Hit Rate, MRR, Precision@K).

Calculate answer quality metrics (ROUGE-L, BLEU, Cosine Similarity).

Save per-chunk size detailed results in testresults.json.

Output
The testresults.json file contains detailed results per question per chunk size including retrieval stats, answers, and metrics.

Summary statistics including averages per chunk size get printed to the console.

Customization & Troubleshooting
Modify chunk sizes in the CHUNK_SIZES dictionary inside evaluation.py.

Adjust TOP_K to change how many chunks are retrieved per query.

Ensure your Ollama server with Mistral model is running locally at port 11434.

API errors or missing servers will result in error messages but do not stop overall execution.

Modify or extend evaluation metrics as needed.

