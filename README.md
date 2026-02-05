# PaperLens RAG: Evidence-First Question Answering for Scientific Papers (Streamlit + FAISS)

PaperLens RAG is a hybrid Retrieval-Augmented Generation (RAG) system designed specifically for scientific publications.
It enables grounded question answering over uploaded research papers while avoiding hallucinations, handling PDF noise, and gracefully degrading when LLMs are unavailable.

This project was built to study RAG deeply, not just demonstrate it.

## Motivation

Scientific PDFs are one of the hardest inputs for RAG systems:
- Key ideas are spread across sections
- Figures, tables, and appendices break text flow
- PDF extraction merges captions and prose
- Definitions are rarely written in one sentence
- Naive RAG systems hallucinate or surface noise

PaperLens RAG addresses these challenges explicitly.

## Core Design Philosophy

- Extractive RAG retrieves evidence.
- Generative RAG synthesizes answers.
- The system must never pretend otherwise.

This project enforces that separation throughout the pipeline.

## System Overview

The system follows a two-mode RAG architecture:

- Extractive RAG → Evidence retrieval (default, safe, no hallucination)
- Generative RAG → Answer synthesis (optional, LLM-powered)

Both modes use the same retrieval pipeline.

## System Architecture Diagram

flowchart TD
    U[User<br/>Streamlit UI]
    U -->|Upload PDF| P[PDF Processing]
    P -->|Extract Pages| C[Chunking<br/>Overlapping Windows]
    C --> E[Embedding Layer]
    E --> V[FAISS Vector Index]
    V --> R[Top-K Retrieval]

    R --> F[Hard Filtering Layer<br/>Headers, Captions, Tokens]
    F --> D[Answer Decision Logic]

    D -->|Evidence| X[Extractive RAG<br/>Cited Evidence]
    D -->|Synthesis| G[Generative RAG<br/>LLM + Citations]

    G -->|Quota / Error| X

## RAG Pipeline (Step-by-Step)

### 1️. PDF Ingestion

- Upload one or more research papers
- Pages are extracted with page numbers preserved
- Reference/bibliography pages are skipped to reduce noise

Why:
Citations must be verifiable and traceable to page numbers.

### 2. Chunking Strategy

Pages are split into overlapping character chunks:

| Parameter        | Value (Typical) | Purpose |
|------------------|-----------------|---------|
| `chunk_chars`    | ~1200 characters | Preserves semantic context within a chunk |
| `overlap_chars`  | ~200 characters  | Prevents information loss at chunk boundaries |
| `min_chars`      | ~250 characters  | Avoids indexing very small or noisy chunks |

This chunking strategy balances retrieval accuracy and context preservation,
which is critical for scientific PDFs where definitions and explanations
often span multiple paragraphs.

Why not sentence chunking?
Scientific PDFs break sentence boundaries due to equations, figures, and formatting.

### 3. Retrieval Strategy (Top-K Similarity Search)

The system retrieves the top-K most semantically similar chunks from the vector index
before applying hard filtering and answer selection.

| Parameter | Typical Range | Purpose |
|----------|--------------|---------|
| `top_k`  | 3 – 10        | Controls how many relevant chunks are retrieved per question |

The Top-K value is user-configurable in the UI, allowing interactive tuning
of retrieval depth for different document types and question complexity.

Over-retrieval is intentional: it ensures high recall, while the hard filtering layer
removes structural noise such as captions, headers, and token dumps before scoring.
This mirrors production RAG systems where recall is prioritized first and precision is enforced downstream.

#### Design trade-offs:
- Lower values improve speed but may miss supporting evidence
- Higher values improve recall but introduce more noise
- The system intentionally over-retrieves and then applies hard filtering

### 4. Embedding & Retrieval

- Chunks are embedded and indexed using FAISS
- Similarity search retrieves top-K chunks
- Over-retrieval is intentional to allow safe filtering later

## Noise Control (Critical Component)

Scientific PDFs contain high-scoring but useless text.

### Explicitly Excluded Content (when headers are OFF)

- Figure captions (Figure 1: …)
- Table titles (Table 5: …)
- Section headers (3.1 Pre-training BERT)
- Appendix headers (A.5 Additional Experiments)
- Diagram labels (TokM, SQuAD, NERMNLI)
- Token dumps (E[CLS], E[SEP], ##)
- Short non-prose fragments (< 10 words)

### Hard Filtering (not penalties)

Captions and headers are removed before scoring, ensuring they never compete with real prose.
**Note:** Only if captions & headers are disabled.

## Extractive RAG Mode (Default)

What it does : 
- Retrieves verbatim sentences from the paper
- Selects question-relevant evidence
- Always includes citations ([Paper.pdf p.X])
- Works without any LLM or API key

What it does NOT do : 
- Does not summarize
- Does not infer
- Does not hallucinate

Guarantee :
“This is the most relevant supporting evidence from the paper.”

## Generative RAG Mode (Optional)

Add the open AI API key in the .env file.
When used : 
- User enables LLM in the UI
- Question requires synthesis (definitions, summaries, comparisons)

Behavior :
- Uses retrieved evidence as context
- LLM is constrained to answer only from sources
- Citations are mandatory

Failure handling, If the LLM:
- Has no API key
- Hits quota limits
- Fails unexpectedly

The system automatically falls back to extractive RAG with a clear UI message.

## Question-Aware Mode Switching (To be done by User)

The system detects question intent:

| Question Type                              | Recommended Mode (User Toggle) |
|-------------------------------------------|---------------------------------|
| Where does the paper discuss … ?           | Extractive (LLM toggle OFF)     |
| Which section introduces … ?               | Extractive (LLM toggle OFF)     |
| Which figure/table summarizes … ?          | Extractive (LLM toggle OFF) + enable headers/captions if needed |
| What are the objectives?                   | Generative (LLM toggle ON)      |
| What is the main contribution?             | Generative (LLM toggle ON)      |
| Summarize the paper                        | Generative (LLM toggle ON)      |
| Compare this method with prior work        | Generative (LLM toggle ON)      |

**Note:** The answer mode is user-controlled. Turn the **LLM toggle ON** to enable generative answers; otherwise the system uses **extractive evidence retrieval** with citations.

## Header & Caption Toggle

Users can choose to:
- Exclude headers & captions → clean prose
- Include headers & captions → structural navigation

This supports both reading styles.

## Hallucination Prevention

PaperLens RAG:
- Separates evidence from answers
- Uses hard structural filtering
- Avoids guessing when information is missing
- Clearly communicates limitations to the user

## Known Limitation (By Design)

Some questions cannot be answered extractively because the paper never states them explicitly. In such cases:
- Extractive mode shows evidence
- Generative mode synthesizes (if enabled)
- Otherwise, the system remains transparent

This is a design strength, not a weakness.

## What This Project Demonstrates

- Deep understanding of RAG internals
- Real-world PDF noise handling
- Honest UX and fallback design
- Production-grade error handling
- Research-oriented transparency

## One-Line Summary

PaperLens RAG is a hybrid RAG system that prioritizes grounded evidence retrieval for scientific PDFs, with optional LLM-based synthesis and explicit hallucination prevention.


## Project File Structure
```text
paperlens-rag-/
  app.py
  rag.py
  eval.py
  requirements.txt
  README.md
```

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```