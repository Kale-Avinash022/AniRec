# Step 20: Embedding Model Selection & Benchmarking

## 1. Process & Purpose
Evaluate embedding models for semantic anime search, comparing inference speed, dimensional footprint, token window, and financial cost.

---

## 2. Comparative Benchmark Matrix

| Model | Provider | Dim | Sequence Length | Speed (CPU) | Cost / 1M Tokens | Verdict |
|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | SentenceTransformers (Local) | 384 | 256 / 512 | ~180 docs/sec | **$0.00** | ✅ **Default Local Pick** (Fast, zero cost, local privacy) |
| `all-mpnet-base-v2` | SentenceTransformers (Local) | 768 | 512 | ~45 docs/sec | **$0.00** | Higher quality, but 4x heavier |
| `text-embedding-3-small` | OpenAI Cloud API | 1536 (or 512) | 8191 | N/A (Cloud) | $0.02 | ✅ **Supported Cloud Pick** (Plug-and-play) |
| `text-embedding-3-large` | OpenAI Cloud API | 3072 | 8191 | N/A (Cloud) | $0.13 | Overkill for ~3,000 documents |

---

## 3. Dimensionality & Space Requirements
- 3,000 shows × 384 float32 dimensions = $\approx 4.6\text{ MB}$ raw vector footprint.
- ChromaDB indexing overhead: $\approx 15\text{ MB}$ on disk total.
- Fast memory residency: Fits entirely in RAM with sub-5ms cosine similarity lookups.

---

## 4. Deliverables & Acceptance Criteria
- [x] Selection of `all-MiniLM-L6-v2` as zero-cost default and `text-embedding-3-small` as cloud option.
- [x] Memory and disk footprint verified to run easily on local development laptops and free-tier cloud containers.
