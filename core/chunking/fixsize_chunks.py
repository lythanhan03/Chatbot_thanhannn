import logging
import re
from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

loger = logging.getLogger(__name__)


class ProcessData:
    def __init__(self, chuck_size: int = 1000, chuck_overlap: int = 100):
        self.chuck_size = chuck_size
        self.chuck_overlap = chuck_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chuck_size,
            chunk_overlap=self.chuck_overlap,
            length_function=len,
            add_start_index=True
        )

    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load PDF
        """
        try:
            loger.info(f"Loading PDF file: {file_path}")
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            return documents
        except Exception as e:
            loger.error(f"Error loading PDF file {file_path}: {e}")
            raise e

    def split_text(self, documents: List[Document]) -> List[Document]:
        """
        Split text into chunks
        """
        try:
            loger.info("Splitting text into chunks")
            texts = self.text_splitter.split_documents(documents)
            # Filter out empty documents
            texts = [doc for doc in texts if doc.page_content.strip()]
            for text in texts:
                text = text.page_content
                text = re.sub(r"\n", " ", text)
                text = re.sub(r"\s+", " ", text)
                text = text.strip()  # Remove leading and trailing spaces
                text = re.sub(r"([?.!,¿])", r" \1 ", text)
                text = re.sub(r'[" "]+', " ", text)
            loger.info(f"Number of chunks created: {len(texts)}")
            return texts
        except Exception as e:
            loger.error(f"Error splitting text: {e}")
            raise e


if __name__ == "__main__":
    # Example usage
    processor = ProcessData()
    pdf_path = Path("/home/quang/Downloads/Ky thuat xay dung-1718.pdf")
    documents = processor.load_pdf(pdf_path)
    chunks = processor.split_text(documents)
    print(f"Number of chunks created: {len(chunks)}")
    print(chunks[:1])