import os
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import requests

# Ensure NLTK data is downloaded
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# SETTINGS
CORPUS_DIR = "corpus"
TEST_DATASET_FILE = "test_dataset.json"
CHUNK_SIZES = {
    "small": 250,
    "medium": 550,
    "large": 900
}
TOP_K = 3

# LOAD TEST DATASET
def load_test_data():
    """Load test dataset with error handling"""
    try:
        with open(TEST_DATASET_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Debug: print the structure
            print(f"Debug: JSON keys found: {list(data.keys())}")
            
            # Try different possible key names
            if 'testquestions' in data:
                questions = data['testquestions']
            elif 'test_questions' in data:
                questions = data['test_questions']
            elif 'questions' in data:
                questions = data['questions']
            elif isinstance(data, list):
                # If the root is already a list
                questions = data
            else:
                print(f"Error: Could not find questions in JSON. Available keys: {list(data.keys())}")
                print(f"Please ensure your JSON has one of these structures:")
                print('  {"testquestions": [...]}')
                print('  {"test_questions": [...]}')
                print('  {"questions": [...]}')
                print('  or just a list: [...]')
                return []
            
            print(f"Debug: Found {len(questions)} questions")
            return questions
            
    except FileNotFoundError:
        print(f"Error: {TEST_DATASET_FILE} not found!")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: {TEST_DATASET_FILE} is not valid JSON!")
        print(f"JSON Error: {e}")
        return []
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []

# LOAD DOCUMENTS
def load_corpus():
    """Load all documents from corpus directory"""
    docs = []
    if not os.path.exists(CORPUS_DIR):
        print(f"Error: {CORPUS_DIR} directory not found!")
        return docs
    
    for fname in sorted(os.listdir(CORPUS_DIR)):
        fpath = os.path.join(CORPUS_DIR, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath, encoding='utf-8') as f:
                    docs.append({'filename': fname, 'text': f.read()})
            except Exception as e:
                print(f"Warning: Could not read {fname}: {e}")
    return docs

# CHUNKING FUNCTION
def chunk_text(text, size):
    """Split text into chunks of specified size"""
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
    """Build embedding index from documents"""
    embeddings, metadatas = [], []
    for doc in docs:
        for chunk in chunk_text(doc['text'], chunk_size):
            embeddings.append(model.encode(chunk))
            metadatas.append({'source': doc['filename'], 'text': chunk})
    return embeddings, metadatas

def run_llm(question, context):
    """
    Calls local Ollama server to get Mistral 7B LLM output.
    context: list of dicts, each with 'text' fields for context chunks.
    Returns: the generated answer from the LLM.
    """
    try:
        # Concatenate retrieved chunks for prompt context
        context_text = "\n\n".join([c['text'] for c in context])
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': 'mistral', 'prompt': prompt, 'stream': False},
            timeout=120  # Add timeout
        )
        response.raise_for_status()  # Raise exception for bad status codes
        resp_json = response.json()
        return resp_json.get('response', '').strip()
    except requests.exceptions.ConnectionError:
        print("Warning: Could not connect to Ollama server. Make sure it's running on localhost:11434")
        return "Error: LLM server not available"
    except Exception as e:
        print(f"Warning: LLM generation failed: {e}")
        return f"Error: {str(e)}"

# RETRIEVAL
def retrieve_chunks(question, index_embeddings, metadatas, model, top_k=TOP_K):
    """Retrieve top-k most relevant chunks for a question"""
    question_vector = model.encode(question)
    sims = cosine_similarity([question_vector], index_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [metadatas[i] for i in top_indices], sims[top_indices]

# EVALUATION STEP
def evaluate_assignment2():
    """Main evaluation function"""
    print("Loading corpus...")
    docs = load_corpus()
    if not docs:
        print("Error: No documents loaded. Exiting.")
        return False
    
    print(f"Loaded {len(docs)} documents")
    
    print("Loading test data...")
    test_data = load_test_data()
    if not test_data:
        print("Error: No test data loaded. Exiting.")
        return False
    
    print(f"Loaded {len(test_data)} test questions")
    
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_results = {}
    
    # Initialize ROUGE scorer and BLEU smoothing
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1  # Add smoothing for BLEU

    for name, csize in CHUNK_SIZES.items():
        print(f"\n{'='*60}")
        print(f"Evaluating for chunk size: {name} ({csize} chars)")
        print(f"{'='*60}")
        
        print("Building embedding index...")
        index_embeddings, metadatas = build_embedding_index(docs, csize, model)
        print(f"Created {len(index_embeddings)} chunks")
        
        results = []

        for i, qa in enumerate(test_data, 1):
            print(f"\nProcessing question {i}/{len(test_data)}: {qa.get('id', 'N/A')}")
            
            qid = qa.get('id', f'q{i}')
            question = qa.get('question', '')
            groundtruth = qa.get('groundtruth', '')
            answerable = qa.get('answerable', True)
            sourcedoc = qa.get('source', '')

            # Retrieval
            top_chunks, sims = retrieve_chunks(question, index_embeddings, metadatas, model)
            retrieved_docs = [c['source'] for c in top_chunks]
            
            # Calculate retrieval metrics
            hit = int(sourcedoc in retrieved_docs) if sourcedoc else 0
            rank = (retrieved_docs.index(sourcedoc) + 1) if sourcedoc in retrieved_docs else 0
            precision_k = sum([1 for d in retrieved_docs if d == sourcedoc]) / TOP_K if sourcedoc else 0

            # Answer generation
            print(f"  Generating answer...")
            answer = run_llm(question, top_chunks)

            # Metrics - Answer Quality: ROUGE-L, BLEU, Cosine Similarity
            if groundtruth and answer:
                rougeL = rouge.score(answer, groundtruth)['rougeL'].fmeasure
                
                # Use smoothing for BLEU to avoid zero scores
                bleu = sentence_bleu(
                    [groundtruth.split()], 
                    answer.split(), 
                    smoothing_function=smoothing
                )
                
                cos_sim = cosine_similarity(
                    [model.encode(answer)], 
                    [model.encode(groundtruth)]
                )[0][0]
            else:
                rougeL = 0.0
                bleu = 0.0
                cos_sim = 0.0

            results.append({
                'question_id': qid,
                'retrieval_metrics': {
                    'hit_rate': hit,
                    'reciprocal_rank': 1 / rank if rank else 0,
                    'precision_k': precision_k
                },
                'answer_metrics': {
                    'rougeL': float(rougeL),
                    'bleu': float(bleu),
                    'cosine_similarity': float(cos_sim)
                }, 
                'retrieved_docs': retrieved_docs,
                'answer': answer,
                'groundtruth': groundtruth
            })
            
            print(f"  Hit: {hit}, RR: {1/rank if rank else 0:.3f}, P@K: {precision_k:.3f}")

        all_results[name] = results
        
        # Calculate and print averages for this chunk size
        avg_hit = sum(r['retrieval_metrics']['hit_rate'] for r in results) / len(results)
        avg_rr = sum(r['retrieval_metrics']['reciprocal_rank'] for r in results) / len(results)
        avg_pk = sum(r['retrieval_metrics']['precision_k'] for r in results) / len(results)
        avg_rouge = sum(r['answer_metrics']['rougeL'] for r in results) / len(results)
        avg_bleu = sum(r['answer_metrics']['bleu'] for r in results) / len(results)
        avg_cos = sum(r['answer_metrics']['cosine_similarity'] for r in results) / len(results)
        
        print(f"\n{name.upper()} CHUNK SIZE AVERAGES:")
        print(f"  Hit Rate: {avg_hit:.3f}")
        print(f"  MRR: {avg_rr:.3f}")
        print(f"  P@{TOP_K}: {avg_pk:.3f}")
        print(f"  ROUGE-L: {avg_rouge:.3f}")
        print(f"  BLEU: {avg_bleu:.3f}")
        print(f"  Cosine Sim: {avg_cos:.3f}")

    # Save results file
    print(f"\n{'='*60}")
    print("Saving results to testresults.json...")
    try:
        with open('testresults.json', 'w', encoding='utf-8') as fout:
            json.dump(all_results, fout, indent=2, ensure_ascii=False)
        print("Results saved successfully!")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

if __name__ == "__main__":
    print("Starting RAG Evaluation...")
    print(f"Configuration:")
    print(f"  Corpus Directory: {CORPUS_DIR}")
    print(f"  Test Dataset: {TEST_DATASET_FILE}")
    print(f"  Chunk Sizes: {CHUNK_SIZES}")
    print(f"  Top-K: {TOP_K}")
    print()
    
    success = evaluate_assignment2()
    if success:
        print("\n✓ Evaluation complete! Check 'testresults.json' for detailed results.")
    else:
        print("\n✗ Evaluation failed. Please check the errors above.")