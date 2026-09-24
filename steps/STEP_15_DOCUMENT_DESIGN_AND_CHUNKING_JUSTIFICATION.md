# Step 15: Single-Doc-per-Show Paradigm vs Chunking Justification

## 1. Process & Purpose
The #1 RAG design decision in candidate interviews is justifying the chunking strategy. While traditional document RAG splits documents into arbitrary 500-token chunks, recommendation systems require a **"Single-Document-Per-Entity"** paradigm. This step documents the theoretical and practical reasons for this architecture.

---

## 2. Comparative Analysis Matrix

| Approach | Working Mechanism | Consequence in Recommendation RAG | Verdict |
|---|---|---|---|
| **Fixed Token Chunking (e.g. 500 tokens)** | Splits synopsis across chunk boundaries; isolates tags from genres | A query matching a plot twist chunk loses the show's title, release year, format, and aggregate score metadata | ❌ **Rejected** (Destroys holistic entity identity) |
| **Sentence-Level Chunking** | Splits each synopsis sentence into individual vectors | Multi-sentence thematic arcs fragmented; scores and tags lost | ❌ **Rejected** (High noise, loss of show context) |
| **Curated Whole-Show Document Synthesis** | Assembles one rich document per show: Title + Year + Genres + Weighted Ranked Tags + Synopsis + Studio | Semantic vector represents the show's entire thematic fingerprint while retaining atomic metadata | ✅ **Selected** (Preserves entity integrity) |

---

## 3. Justification Narrative for Technical Writeup & Interviews
> *"In unstructured document retrieval (e.g. PDF manuals), chunking is needed because passages answer isolated questions. However, anime recommendation is a holistic entity-level judgment. Chunking a show's synopsis into paragraphs destroys the vital connection between its community tags (e.g. 'dark fantasy (85)'), genre classifications, production studio, and overall plot theme. By engineering a unified document representation per show, our vector embedding captures the exact semantic fingerprint of the entire media entity."*

---

## 4. Deliverables & Acceptance Criteria
- [x] Clear theoretical justification for Single-Doc-per-Show strategy documented.
- [x] Architecture ready to feed into Document Synthesizer in Step 16 and Step 17.
