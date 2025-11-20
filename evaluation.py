import os
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu

# Ensure NLTK data is downloaded
import nltk
nltk.download('punkt')

# SETTINGS
CORPUS_DIR = "corpus"
TEST_DATASET_FILE = "testdataset.json"
CHUNK_SIZES = {
    "small": 250,
    "medium": 550,
    "large": 900
}
TOP_K = 3

# LOAD TEST DATASET
with open(TEST_DATASET_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)['testquestions']

# LOAD DOCUMENTS
def load_corpus():
    docs = []
    for fname in sorted(os.listdir(CORPUS_DIR)):
        with open(os.path.join(CORPUS_DIR, fname), encoding='utf-8') as f:
            docs.append({'filename': fname, 'text': f.read()})
    return docs

# CHUNKING FUNCTION
def chunk_text(text, size):
    chunks = []
    idx = 0
    while idx < len(text):
        chunk = text[idx:idx+size]
        if chunk.strip():
            chunks.append(chunk)
        idx += size
    return chunks

# EMBEDDINGS FUNCTIONS
def build_embedding_index(docs, chunk_size, model):
    embeddings, metadatas = [], []
    for doc in docs:
        for chunk in chunk_text(doc['text'], chunk_size):
            embeddings.append(model.encode(chunk))
            metadatas.append({'source': doc['filename'], 'text': chunk})
    return embeddings, metadatas

import requests

def run_llm(question, context):
    """
    Calls local Ollama server to get Mistral 7B LLM output.
    context: list of dicts, each with 'text' fields for context chunks.
    Returns: the generated answer from the LLM.
    """
    # Concatenate retrieved chunks for prompt context
    context_text = "\n\n".join([c['text'] for c in context])
    prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"

    response = requests.post(
        'http://localhost:11434/api/generate',
        json={'model': 'mistral', 'prompt': prompt, 'stream': False}
    )
    resp_json = response.json()
    return resp_json.get('response', '').strip()


# RETRIEVAL
def retrieve_chunks(question, index_embeddings, metadatas, model, top_k=TOP_K):
    question_vector = model.encode(question)
    sims = cosine_similarity([question_vector], index_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [metadatas[i] for i in top_indices], sims[top_indices]

# EVALUATION STEP
def evaluate_assignment2():
    docs = load_corpus()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_results = {}

    for name, csize in CHUNK_SIZES.items():
        print(f"Evaluating for chunk size: {name}")
        index_embeddings, metadatas = build_embedding_index(docs, csize, model)
        results = []

        for qa in test_data:
            qid = qa['id']
            question = qa['question']
            groundtruth = qa['groundtruth']
            answerable = qa['answerable']
            sourcedoc = qa.get('source', '')

            # Retrieval
            top_chunks, sims = retrieve_chunks(question, index_embeddings, metadatas, model)
            retrieved_docs = [c['source'] for c in top_chunks]
            hit = int(sourcedoc in retrieved_docs) if sourcedoc else 0
            rank = (retrieved_docs.index(sourcedoc)+1) if sourcedoc in retrieved_docs else 0
            precision_k = sum([1 for d in retrieved_docs if d == sourcedoc]) / TOP_K if sourcedoc else 0

            # Answer generation
            answer = run_llm(question, top_chunks)

            # Metrics
            # Answer Quality: ROUGE-L, BLEU, Cosine Similarity
            rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rougeL = rouge.score(answer, groundtruth)['rougeL'].fmeasure
            bleu = sentence_bleu([groundtruth.split()], answer.split())
            cos_sim = cosine_similarity([model.encode(answer)], [model.encode(groundtruth)])[0][0]

            results.append({
                'question_id': qid,
                'retrieval_metrics': {
                    'hit_rate': hit,
                    'reciprocal_rank': 1 / rank if rank else 0,
                    'precision_k': precision_k
                },
                'answer_metrics': {
                    'rougeL': rougeL,
                    'bleu': bleu,
                    'cosine_similarity': cos_sim
                }, 
                'retrieved_docs': retrieved_docs,
                'answer': answer,
                'groundtruth': groundtruth
            })

        all_results[name] = results

    # Save results file
    with open('testresults.json', 'w', encoding='utf-8') as fout:
        json.dump(all_results, fout, indent=2)

if __name__ == "__main__":
    evaluate_assignment2()
    print("Evaluation complete. Check 'testresults.json'.")

