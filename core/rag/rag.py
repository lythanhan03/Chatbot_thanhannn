from langchain_core.documents import Document
from typing import List, TypedDict
from core.chunking.fixsize_chunks import ProcessData
from core.embeding.embedings import VectorStore
from core.llm.gemini_llm import LLM


document_path = "/home/quang/Downloads/Ky thuat xay dung-1718.pdf"
processor = ProcessData()
documents = processor.load_pdf(document_path)
chunks = processor.split_text(documents)
print("Text chunks created successfully")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

vector_store = VectorStore().create_vector_db(chunks)
print("Vector store created and saved successfully")

def query(state: State, k : int = 3):
    retrieved_docs = vector_store.similarity_search(state["question"], k=k)
    return {**state,
        "context": retrieved_docs}

def generate(state: State):
    llm = LLM()
    prompt = llm.get_query_prompt(state['question'])
    print("Generated prompt:", prompt) 
    answer = llm.post_request(prompt)
    return {
        **state,
        "answer": answer
    }

def run(state: State):
    state = query(state)
    state = generate(state)
    return state

if __name__ == "__main__":
    initial_state: State = {
        "question": "muc tieu cu the cua ky thuat xay dung la gi ?",
        "context": [],
        "answer": ""
    }
    final_state = run(initial_state)
    print("", final_state["answer"])