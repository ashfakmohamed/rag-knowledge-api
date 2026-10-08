from pathlib import Path

from app.retriever import StoredDocument, TfidfRetriever

CASES = [
    ("How is the portfolio deployed?", "portfolio"),
    ("How should Django secrets be configured?", "engineering"),
]


def main() -> None:
    retriever = TfidfRetriever()
    knowledge_directory = Path(__file__).resolve().parents[1] / "knowledge"
    for path in knowledge_directory.glob("*.md"):
        retriever.add(
            StoredDocument(
                id=path.stem,
                title=path.stem.replace("-", " ").title(),
                content=path.read_text(encoding="utf-8"),
                source=str(path),
            )
        )

    hits = 0
    for question, expected_id in CASES:
        results = retriever.search(question, top_k=1)
        predicted_id = results[0][0].id if results else None
        hits += int(predicted_id == expected_id)
        print(f"question={question!r} expected={expected_id!r} predicted={predicted_id!r}")

    recall_at_one = hits / len(CASES)
    print(f"recall@1={recall_at_one:.2f}")
    if recall_at_one < 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
