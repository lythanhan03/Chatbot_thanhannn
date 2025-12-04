from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config.config import *


class Semantic_Chunk:
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME_EMBEDDING)

    def load_pdf(self) -> List[Document]:
        loader = PyPDFLoader(self.document_path)
        documents = loader.load()
        return documents

    def semantic_chunk(self, documents: List[Document]) -> List[Document]:
        raw_texts = [doc.page_content for doc in documents]
        chunker = SemanticChunker(embeddings=self.embeddings,
                                  breakpoint_threshold_type="percentile",
                                  breakpoint_threshold_amount=80)
        chunks = chunker.create_documents(raw_texts)
        return chunks


if __name__ == "__main__":
    document_path = Path("/home/quang/Downloads/Ky thuat xay dung-1718.pdf")
    processor = Semantic_Chunk(document_path=document_path)
    documents = processor.load_pdf()
    chunks = processor.semantic_chunk(documents)
    print(f"Number of chunks created: {len(chunks)}")
    print(chunks[:1])