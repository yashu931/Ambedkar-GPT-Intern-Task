import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


DOCUMENT_PATH = "speech.txt"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_and_process_documents(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    print(f"Loading document from: {file_path}")
    loader = TextLoader(file_path)
    documents = loader.load()

    print("Splitting document into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")
    return chunks


def setup_rag_pipeline(chunks):
    print(f"Loading HuggingFace Embeddings: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(CHROMA_DB_DIR):
        print(f"Loading existing Chroma store: {CHROMA_DB_DIR}")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
    else:
        print(f"Creating Chroma vector store at {CHROMA_DB_DIR}")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        vectorstore.persist()

    retriever = vectorstore.as_retriever()

    print("Initializing Ollama (mistral)")
    llm = Ollama(model="mistral")

    # Prompt Template
    prompt = ChatPromptTemplate.from_template("""
        You are an assistant answering ONLY using the provided context.
        
        CONTEXT:
        {context}

        QUESTION:
        {question}

        Provide the most relevant answer.
    """)

    # NEW LCEL RAG PIPELINE
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain


def main():
    try:
        chunks = load_and_process_documents(DOCUMENT_PATH, chunk_size=200, chunk_overlap=0)
        rag_chain = setup_rag_pipeline(chunks)

        print("\n--- AmbedkarGPT Ready ---")
        print("Ask a question based on speech.txt")
        print("Type 'exit' to quit.\n")

        while True:
            question = input("Your Question: ")

            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            print("Thinking...")
            answer = rag_chain.invoke(question)
            print("\n## Answer\n")
            print(answer)
            print("\n")

    except Exception as e:
        print(f"\nError: {e}")
        if "Failed to connect" in str(e):
            print("Make sure Ollama is running and 'mistral' is installed.")


if __name__ == "__main__":
    main()
