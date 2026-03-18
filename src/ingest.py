import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def ingest_documents(docs_path: str = "./docs", persist_path: str = "./chroma_db"):
    print("Loading documents...")
    loaders = [
        DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader),
    ]

    raw_docs = []
    for loader in loaders:
        try:
            raw_docs.extend(loader.load())
        except Exception as e:
            print(f"Loader warning: {e}")

    if not raw_docs:
        print("No documents found in docs/ folder. Add some PDF or TXT files first.")
        return

    print(f"Loaded {len(raw_docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks")

    print("Embedding and indexing...")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path
    )

    print(f"Done. {len(chunks)} chunks indexed into {persist_path}")
    return vectorstore

if __name__ == "__main__":
    ingest_documents()
