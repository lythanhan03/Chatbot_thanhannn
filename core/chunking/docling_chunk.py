from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from config.config import *


class DoclingChunk():
    def __init__(self, model_name: str = MODEL_NAME_EMBEDDING, file_path: str = DOCUMENT_PATH):
        self.embed_model_id = model_name
        self.file_path = file_path

    def load_docling(self):
        """Load documents from Docling."""
        EXPORT_TYPE = ExportType.DOC_CHUNKS
        loader = DoclingLoader(file_path=self.file_path,
                               export_type=EXPORT_TYPE,
                               chunker=HybridChunker(tokenizer=self.embed_model_id))
        documents = loader.load()
        return documents


if __name__ == "__main__":
    doc = DoclingChunk().load_docling()
    print(doc[:3])