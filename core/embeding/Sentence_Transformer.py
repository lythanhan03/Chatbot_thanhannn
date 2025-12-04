from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from core.embeding.base import BaseEmbedding
from langchain_core.documents import Document
from pyvi.ViTokenizer import tokenize


class SentenceEmbed(BaseEmbedding):
    def __init__(self, model_name: str = "dangvantuan/vietnamese-embedding"):
        self.embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        self.vector_db = None

    def create_vector_store(self, documents: Document) -> Chroma:
        """Create vector store from documents."""
        documents = [Document(page_content=text) for text in documents]
        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        return self.vector_db

    def save_vector_store(self, path: str = "vectordb") -> None:
        """Save vector store to specified path."""
        if self.vector_db is not None:
            self.vector_db.save_local(path)

    def load_vector_store(self, path: str = "vectordb") -> None:
        """Load vector store from specified path."""
        self.vector_db = Chroma.load_local(path, self.embeddings)
        return self.vector_db


if __name__ == "__main__":
    documents = "Những bài văn, đoạn văn mẫu lớp 2 sách mới Kết nối tri thức, Chân trời sáng tạo, Cánh diều hay nhất, chọn lọc từ những đoạn văn hay của học sinh lớp 2 trên cả nước giúp học sinh luyện viết đoạn văn lớp 2 hay hơn."
    processor = SentenceEmbed()
    vector_store = processor.create_vector_store(documents)
    print(f"Vector store created")
    vector_store.save_vector_store()
    print("Vector store saved successfully")