from dataclasses import dataclass
from threading import RLock

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class StoredDocument:
    id: str
    title: str
    content: str
    source: str


class TfidfRetriever:
    """Small deterministic retriever suitable for local demos and tests."""

    def __init__(self) -> None:
        self._documents: dict[str, StoredDocument] = {}
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = None
        self._ordered_documents: list[StoredDocument] = []
        self._lock = RLock()

    def add(self, document: StoredDocument) -> None:
        with self._lock:
            self._documents[document.id] = document
            self._ordered_documents = list(self._documents.values())
            corpus = [f"{item.title} {item.content}" for item in self._ordered_documents]
            self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3) -> list[tuple[StoredDocument, float]]:
        with self._lock:
            if self._matrix is None or not self._ordered_documents:
                return []
            query_vector = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self._matrix)[0]
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
            return [
                (self._ordered_documents[index], float(score))
                for index, score in ranked[:top_k]
                if score > 0
            ]

    @property
    def count(self) -> int:
        return len(self._documents)
