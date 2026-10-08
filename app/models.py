from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=20, max_length=50_000)
    source: str = Field(min_length=1, max_length=500)


class DocumentResponse(DocumentCreate):
    id: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)
    top_k: int = Field(default=3, ge=1, le=10)


class Citation(BaseModel):
    document_id: str
    title: str
    source: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[Citation]
