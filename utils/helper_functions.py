from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

# Explicitly define everything that should be available through
# ``from utils.helper_functions import *``.
__all__ = [
    "ChatOpenAI",
    "OpenAIEmbeddings",
    "RecursiveCharacterTextSplitter",
    "FAISS",
    "PyPDFLoader",
    "PromptTemplate",
    "replace_tab_with_space",
    "encode_pdf",
    "show_related_docs"
]

def replace_tab_with_space(list_of_documents):
    for doc in list_of_documents:
        doc.page_content=doc.page_content.replace('\t',' ')
    return list_of_documents

def encode_pdf(file_path,chunk_size=1000,chunk_overlap=500):
    loader=PyPDFLoader(file_path)
    docs=loader.load()

    chunker=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks=chunker.split_documents(docs)
    cleaned_chunks=replace_tab_with_space(chunks)
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(cleaned_chunks,embeddings)

    return vectorstore

def show_related_docs(context):
    for i,c in enumerate(context):
        print(f"Context:{i+1}")
        print(c)
        print('\n')

