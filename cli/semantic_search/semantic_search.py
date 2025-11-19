from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def verify_model() -> None:
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)
