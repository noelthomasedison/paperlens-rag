from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from rag import RAGIndex, chunk_pages, extractive_answer, llm_answer_openai, format_citation, read_pdf_pages

st.set_page_config(page_title="Paper RAG Demo", page_icon="📄", layout="wide")

st.title("📄 Paper RAG Demo — Chat with PDFs (with citations)")
st.caption("Upload research paper PDFs, index them locally, and ask questions with page-level citations.")

with st.sidebar:
    st.header("Index settings")
    chunk_chars = st.slider("Chunk size (chars)", 600, 2000, 1200, 100)
    overlap_chars = st.slider("Overlap (chars)", 0, 500, 200, 50)
    top_k = st.slider("Top-K retrieval", 3, 12, 5, 1)
    use_llm = st.toggle("Use LLM if configured (OPENAI_API_KEY)", value=False)
    include_refs = st.toggle("Include References section", value=False)
    include_headers = st.toggle("Include section headers & figure captions",value=False,help="Enable this to retrieve section titles, figure captions, and table headers.")

    st.divider()
    st.header("Upload PDFs")
    uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "docs" not in st.session_state:
    st.session_state.docs = []

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("1) Index documents")
    if st.button("Build / Rebuild Index", type="primary", disabled=not uploaded):
        all_chunks = []
        docs = []
        for f in uploaded:
            pages = read_pdf_pages(f.read())
            docs.append((f.name, len(pages)))
            all_chunks.extend(chunk_pages(f.name, pages, chunk_chars=chunk_chars, overlap_chars=overlap_chars))

        rag = RAGIndex(model_name="all-MiniLM-L6-v2")
        if not all_chunks:
            st.error("No text extracted. If your PDF is scanned, you’ll need OCR.")
        else:
            rag.build(all_chunks)
            st.session_state.rag_index = rag
            st.session_state.chunks = all_chunks
            st.session_state.docs = docs
            st.success(f"Indexed {len(all_chunks)} chunks from {len(docs)} PDF(s).")

    if st.session_state.docs:
        st.write("**Indexed documents:**")
        for name, n_pages in st.session_state.docs:
            st.write(f"- {name} ({n_pages} pages)")
            
def is_definition_or_summary_question(q: str) -> bool:
        q = q.lower()
        return any(
            phrase in q
            for phrase in [
                "what are the",
                "what is the",
                "summarize",
                "summary",
                "main contributions",
                "objectives",
                "advantages",
                "limitations",
                "key findings",
            ]
        )

with colB:
    st.subheader("2) Ask questions")
    question = st.text_input("Try: “What are the main contributions?” / “What dataset is used?” / “What are limitations?”")

    if st.button("Search & Answer", disabled=(not question or not st.session_state.rag_index)):
        rag: RAGIndex = st.session_state.rag_index
        retrieved = rag.search(question, top_k=top_k+5)
        if not include_refs:
            retrieved = [(c, s) for (c, s) in retrieved if not c.is_references]
        retrieved = retrieved[:top_k]

        st.markdown("### ✅ Answer")
        # 🔹 Auto-switch: use LLM mainly for "summary/definition/list" style questions
        gen_triggers = [
            "summarize", "summary", "main contribution", "contribution",
            "what are the", "what is the", "explain", "define",
            "objectives", "limitations", "future work", "compare",
        ]
        should_generate = any(t in question.lower() for t in gen_triggers)
        if is_definition_or_summary_question(question):
            st.caption(
                "This question requires synthesizing information across sections. "
                "Extractive mode shows supporting evidence; generative mode produces a concise answer."
            )
            
        if use_llm and should_generate:
            ans, status, detail = llm_answer_openai(question, retrieved)
            if status == "OK":
                pass  # ans already set
            elif status == "NO_KEY":
                st.info("LLM not configured (missing OPENAI_API_KEY). Showing extractive RAG evidence.")
                ans = extractive_answer(question, retrieved, include_headers=include_headers)
            elif status == "QUOTA":
                st.warning("LLM unavailable (quota or rate limit). Falling back to extractive RAG evidence.")
                with st.expander("Error details (for debugging)"):
                    st.code(detail or "")
                ans = extractive_answer(question, retrieved, include_headers=include_headers)
            elif status == "ERROR":
                st.error("LLM failed due to an unexpected error. Falling back to extractive RAG evidence.")
                with st.expander("Error details (for debugging)"):
                    st.code(detail or "")
                ans = extractive_answer(question, retrieved, include_headers=include_headers)
        else:
            ans = extractive_answer(question, retrieved, include_headers=include_headers)

        if not include_headers:
            st.caption("Headers, figure captions, and table titles are excluded for cleaner evidence.")
        else:
            st.caption("Headers and captions are included in retrieval results.")

        st.write(ans)

        st.markdown("### 🔎 Citations")
        if retrieved:
            cites = []
            for c, _score in retrieved:
                cites.append(f"- [{format_citation(c)}]")
            st.write("\n".join(cites))
        else:
            st.write("No relevant chunks retrieved.")

        st.markdown("### 📌 Retrieved context (top matches)")
        for i, (c, score) in enumerate(retrieved, start=1):
            with st.expander(f"{i}. {format_citation(c)} — score {score:.3f}"):
                st.write(c.text)
