import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector embeddings and database operations."""

    def __init__(self, model_name: str = "AITeamVN/Vietnamese_Embedding"):  # "dangvantuan/vietnamese-embedding"
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_db = None

    def create_vector_db(self, documents: Document) -> FAISS:
        """Create vector database from documents."""

        try:
            logger.info("Creating vector database")
            self.vector_db = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            return self.vector_db
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete vector database collection."""
        if self.vector_db:
            try:
                logger.info("Deleting vector database collection")
                self.vector_db.delete_collection()
                self.vector_db = None
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")
                raise

    def save_vector_db(self, path: str) -> None:
        """Save vector database to specified path."""
        if self.vector_db is not None:
            try:
                logger.info(f"Saving vector database to {path}")
                self.vector_db.save_local(path)
            except Exception as e:
                logger.error(f"Error saving vector database: {e}")
                raise

    def load_vector_db(self, path: str) -> None:
        """Load vector database from specified path."""
        try:
            logger.info(f"Loading vector database from {path}")
            self.vector_db = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            raise