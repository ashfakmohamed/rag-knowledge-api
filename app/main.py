import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI

from app.generator import generator_from_environment
from app.models import Citation, DocumentCreate, DocumentResponse, QueryRequest, QueryResponse
from app.retriever import StoredDocument, TfidfRetriever

retriever = TfidfRetriever()
generator = generator_from_environment()


def load_knowledge(directory: Path) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        retriever.add(
            StoredDocument(
                id=path.stem,
                title=path.stem.replace("-", " ").title(),
                content=content,
                source=str(path.as_posix()),
            )
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    default_directory = Path(__file__).resolve().parents[1] / "knowledge"
    load_knowledge(Path(os.environ.get("KNOWLEDGE_DIR", default_directory)))
    yield


app = FastAPI(
    title="RAG Knowledge API",
    description="Grounded retrieval and generation with explicit citations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, int | str]:
    return {"status": "ok", "documents": retriever.count}


@app.post("/documents", response_model=DocumentResponse, status_code=201)
async def create_document(payload: DocumentCreate) -> DocumentResponse:
    document = StoredDocument(id=str(uuid4()), **payload.model_dump())
    retriever.add(document)
    return DocumentResponse(id=document.id, **payload.model_dump())


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    results = retriever.search(payload.question, payload.top_k)
    answer = await generator.generate(payload.question, results)
    citations = [
        Citation(
            document_id=document.id,
            title=document.title,
            source=document.source,
            score=round(score, 4),
            excerpt=document.content.strip()[:240],
        )
        for document, score in results
    ]
    return QueryResponse(answer=answer, grounded=bool(citations), citations=citations)
