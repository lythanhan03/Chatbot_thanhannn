import faiss

class QueryFAISS:
    def query_with_faiss(self, doc_vector, query_vector, k=3):
        index = faiss.IndexFlatL2(doc_vector.shape[1])
        index.add(doc_vector)
        distances, indices = index.search(query_vector, k)
        return indices, distances