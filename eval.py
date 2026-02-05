import json
from rag import RAGIndex, chunk_pages, read_pdf_pages

def run_eval(pdf_path: str, eval_json_path: str, top_k: int = 5):
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    pages = read_pdf_pages(file_bytes)
    chunks = chunk_pages(doc_name=pdf_path.split("/")[-1], pages=pages)

    rag = RAGIndex()
    rag.build(chunks)

    with open(eval_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    hits = 0
    for item in items:
        q = item["q"]
        gold_page = int(item["gold_page"])
        retrieved = rag.search(q, top_k=top_k)
        retrieved_pages = [c.page for c, _ in retrieved]
        ok = gold_page in retrieved_pages
        hits += 1 if ok else 0
        print(f"Q: {q}\n  retrieved_pages={retrieved_pages}\n  hit={ok}\n")

    recall = hits / max(1, len(items))
    print(f"Recall@{top_k}: {recall:.2f} ({hits}/{len(items)})")

if __name__ == "__main__":
    # Example:
    # python eval.py
    run_eval("sample_paper.pdf", "eval.json", top_k=5)
