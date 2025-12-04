from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from core.embeding.base import BaseEmbedding
from config.config import *
import logging
import os

logger = logging.getLogger(__name__)


class HuggingEmbed(BaseEmbedding):
    def __init__(self, name: str = MODEL_NAME_EMBEDDING):
        self.embeddings = HuggingFaceEmbeddings(model_name=name)
        self.vector_db = None

    def create_vector_store(self, documents: Document) -> FAISS:
        """Create vector store from documents."""
        self.vector_db = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        return self.vector_db

    def save_vector_store(self, path: str = "vectordb") -> None:
        """Save vector store to specified path."""
        if self.vector_db is not None:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            self.vector_db.save_local(path)
            logger.info(f"Vector store saved successfully to {path}")
        else:
            logger.warning("No vector store to save.")

    def load_vector_store(self, path: str = "vectordb") -> None:
        """Load vector store from specified path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at {path}")
        self.vector_db = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully.")

    def add_documents_to_store(self, documents: Document, path: str = "vectordb") -> None:
        """Add new documents to an existing vector store."""
        if not os.path.exists(path):
            logger.warning(f"Vector store not found at {path}. Creating a new one instead.")
            self.create_vector_store(documents)
        else:
            self.load_vector_store(path)
            logger.info(f"Adding {len(documents)} new chunks to the existing vector store.")
            self.vector_db.add_documents(documents)

        self.save_vector_store(path)
        logger.info("New documents successfully added and vector store saved.")