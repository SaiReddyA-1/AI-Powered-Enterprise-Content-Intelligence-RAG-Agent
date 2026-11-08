# Enterprise Content Intelligence

### AI-Powered RAG Platform for Enterprise Document Intelligence

A production-ready, fully dockerized AI system that ingests enterprise documents of any format, builds a searchable vector knowledge base with intelligent metadata classification, and delivers grounded answers with source citations. Runs entirely locally except LLM inference (Groq API).

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Dashboard Pages](#dashboard-pages)
- [Offline Pipeline Deep-Dive](#offline-pipeline-deep-dive)
- [Online Query Pipeline Deep-Dive](#online-query-pipeline-deep-dive)
- [Incremental Processing & File Manifest](#incremental-processing--file-manifest)
- [Handling Uncategorized Data at Scale](#handling-uncategorized-data-at-scale)
- [Persistent Chat History](#persistent-chat-history)
- [Reliability & Fault Tolerance](#reliability--fault-tolerance)
- [Scalability](#scalability)
- [Amazon Bedrock Migration Path](#amazon-bedrock-migration-path)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Stopping & Restarting](#stopping--restarting)
- [Troubleshooting](#troubleshooting)

---

## Key Capabilities

- **AI-Powered Q&A with Citations** - Ask natural language questions, get answers grounded in your documents with exact source file references and confidence scores
- **Smart Query Routing** - Automatically detects analytics queries ("how many invoices?"), file-specific queries ("show me invoice_10256.pdf"), and semantic search queries
- **Automatic Document Classification** - Triage AI classifies document type, department, and quality score using a 4-tier engine (folder path -> filename -> content regex -> LLM auto-discovery)
- **Incremental Processing** - File manifest tracks every document's state; only new, modified, or deleted files are processed on restart (90s full ingestion -> 4s restart)
- **OCR for Scanned Documents** - Tesseract OCR extracts text from JPG, PNG, TIFF, BMP images and scanned PDFs with automatic page-level rendering
- **Dynamic Category Discovery** - Triage auto-discovers new document types via LLM when content doesn't match known patterns
- **Conversation Memory** - Follow-up questions maintain context; persistent chat history survives browser refresh and container restarts
- **Full Document Retrieval** - When asking about a specific file, ALL chunks are retrieved and sent to the LLM for complete context
- **Multi-Format Support** - PDF, DOCX, Markdown (.md), CSV, XLSX, XLS, TXT, JSON, XML, JPG, JPEG, PNG
- **Page-Aware Chunking** - PDF chunks preserve page numbers for precise citations
- **Folder & Path-Aware Search** - Query by folder ("files from invoices folder"), file extension ("pdf files"), or any metadata attribute
- **4-Page Dashboard** - AI Chat, Data Manager, Pipeline Monitor, Intelligence Dashboard
- **Real-Time Metrics** - All dashboard values are live from the database, not hardcoded
- **In-Memory Fallback** - System remains fully functional even if OpenSearch is unavailable

---

## Architecture

> Full interactive architecture diagram available in [`architecture.mmd`](architecture.mmd) (Mermaid format - render with GitHub, VS Code Mermaid plugin, or [mermaid.live](https://mermaid.live))

```
OFFLINE PIPELINE (runs on startup - incremental via File Manifest)
┌──────────────┐   ┌──────────┐   ┌─────┐   ┌───────────┐   ┌─────────┐   ┌──────────┐   ┌────────────┐
│  Repository  │──>│ Document │──>│ OCR │──>│ Triage AI │──>│ Chunker │──>│ Embedder │──>│ OpenSearch │
│   Scanner    │   │  Parser  │   │     │   │           │   │         │   │          │   │   Index    │
└──────┬───────┘   └──────────┘   └─────┘   └───────────┘   └─────────┘   └──────────┘   └────────────┘
       │            PDF/DOCX/MD   Conditional  Type+Dept+     Page-aware   BAAI/bge-large    HNSW+kNN
       │            per-page text on low text   Quality        recursive     1024 dims        cosine
       v
┌──────────────┐
│ File Manifest│  Tracks new/modified/deleted/unchanged files
│  (JSON)      │  SHA-256 content hashing · Atomic writes
└──────────────┘

ONLINE PIPELINE (per query)
┌────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────┐   ┌─────────┐   ┌───────────────┐
│ User Query │──>│ Intent Parse │──>│ Hybrid Search │──>│ Reranker │──>│  LLM    │──>│ Answer +      │
│            │   │ + Embed      │   │ Vector + Meta  │   │CrossEnc. │   │ (Groq)  │   │ Citations     │
└────────────┘   └──────────────┘   └───────────────┘   └──────────┘   └─────────┘   └───────────────┘
                 Metadata filters   Top 15 candidates    Top 5 best   3-layer context  Source files
                 folder/ext/type    kNN + term filters   query-chunk   unified prompt   + confidence
```

---

## Tech Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| **Vector Database** | OpenSearch 2.11 | HNSW index, cosine similarity, ef_construction=128, m=16 |
| **Embeddings** | BAAI/bge-large-en-v1.5 | 1024-dimensional dense vectors, local inference |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | CrossEncoder for precise query-chunk scoring |
| **LLM** | Groq API (Llama 3.3 70B) | Fast inference, temperature=0.1 for factual answers |
| **UI** | Streamlit | Multi-page dashboard with real-time metrics |
| **OCR** | Tesseract + Pillow | Text extraction from images and scanned PDFs |
| **PDF Parsing** | PyMuPDF (fitz) | Per-page text extraction with image detection |
| **Document Formats** | python-docx, pandas, openpyxl | DOCX, Markdown, CSV, XLSX support |
| **State Tracking** | FileManifest (JSON) | Incremental processing with SHA-256 content hashing |
| **Container** | Docker Compose | OpenSearch + App, single command startup |
| **Language** | Python 3.11 | Type-annotated, Pydantic settings management |

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running (allocate 4GB+ RAM)
- A free [Groq API key](https://console.groq.com/keys)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Navyasri-V/AI-POWERED_ENTERPRISE_CONTENT_INTELLIGANCE_USING_RAG.git
   cd AI-POWERED_ENTERPRISE_CONTENT_INTELLIGANCE_USING_RAG
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Groq API key:
   ```
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

3. **Place your documents**

   Put files in `Data/Rag/`. Subdirectories are optional but improve classification accuracy:
   ```
   Data/Rag/
   ├── invoices/          <- auto-classified as invoice/finance (95% confidence)
   ├── purchase_orders/   <- auto-classified as purchase_order/operations
   ├── shipping_orders/   <- auto-classified as shipping_order/operations
   ├── inventory_reports/ <- auto-classified as inventory_report/operations
   ├── markdown_docs/     <- .md files fully supported
   └── random_dump/       <- still works - triage classifies by filename + content
   ```

4. **Start the application**
   ```bash
   docker-compose up --build
   ```

5. **Open the dashboard**

   Navigate to [http://localhost:8501](http://localhost:8501)

### What Happens on Startup

1. **OpenSearch** starts and creates the HNSW vector index with all metadata mappings
2. **File Manifest** loads from `Data/file_manifest.json` (or creates a new one)
3. **Filesystem diff** identifies new, modified, deleted, and unchanged files
4. **Only changed files** go through: Parse -> OCR -> Triage -> Chunk -> Embed -> Index
5. Deleted files have their chunks removed from OpenSearch
6. Manifest is saved with full metadata for each processed file
7. **Streamlit** starts on port 8501

**First run**: ~5 minutes (downloads ~1.3GB embedding model + ~80MB reranker during Docker build). Subsequent starts skip unchanged files entirely (~4 seconds).

### Why the App Takes a Moment to Load

When you first open `http://localhost:8501`, Streamlit initializes the backend via `@st.cache_resource`:

| Component | Load Time | Size |
|-----------|-----------|------|
| SentenceTransformer (BAAI/bge-large-en-v1.5) | 5-10s | ~1.3GB into RAM |
| CrossEncoder (ms-marco-MiniLM-L-6-v2) | 2-3s | ~80MB into RAM |
| OpenSearch connection | 1-3s | Network handshake |
| Streamlit framework | 2-3s | Python compilation |
| **Total first load** | **10-20s** | **Cached after first visit** |

All subsequent page navigations are instant - models stay in memory for the session lifetime.

---

## Dashboard Pages

### AI Chat

The primary interface for querying your document repository.

**Features:**
- Natural language queries with source citations
- Conversation memory - follow-up questions reference previous context ("tell me more about that invoice")
- Persistent chat history - conversations saved to disk, survive browser refresh and container restarts
- Load, switch, and manage past conversations
- Confidence score badges (green >=80%, yellow 50-79%, red <50%)
- Query type indicators (Analytics, Semantic Search, Full Document)

**Query Types Handled:**

| Query Type | Example | How It Works |
|------------|---------|-------------|
| Analytics | "How many invoices do we have?" | Metadata aggregation, no vector search needed |
| Semantic Search | "Show me finance-related invoices" | Vector similarity + metadata filters + reranking |
| File-Specific | "Tell me about invoice_10256.pdf" | All chunks of that file retrieved for full context |
| Follow-up | "Tell me more about that" | Context from previous response used to expand query |
| Folder Search | "Files from the invoices folder" | Path-based metadata filtering |
| Extension Filter | "Show me all pdf files" | File extension metadata filtering |
| Multi-topic | "How many invoices and what do shipping orders contain?" | Unified 3-layer context handles all sub-questions |

### Data Manager

Browse, search, filter, upload, and delete documents.

**Features:**
- Summary metrics bar: document count, chunk count, type count, department count
- Searchable/filterable document table with progress bars for quality scores
- Upload new documents (including .md) - processed through the full pipeline instantly
- Delete documents and all their chunks from the index
- File manifest automatically updated on upload and delete
- Real-time refresh

### Pipeline Monitor

Visualize every stage of the processing pipeline.

**Features:**
- 7-stage pipeline visualization with live metrics per stage
- Database state: mode (OpenSearch/memory), index name, chunk count, store size
- Triage classification results: bar charts for document types and departments
- Quality distribution: excellent (>=90%), good (70-89%), fair (50-69%), degraded (<50%)
- Active configuration table: all pipeline settings from `.env`

### Intelligence Dashboard

Enterprise analytics on document quality and risk.

**Features:**
- Hero metrics: total documents, total chunks, repository health score (color-coded)
- Distribution charts: documents by department and type
- Degradation report: count, percentage, threshold alerts
- Common issues tracking (noisy OCR, low text, etc.)

---

## Offline Pipeline Deep-Dive

### Stage 1: Repository Scanner
**File:** `services/ingestion/connectors.py`

- Recursively scans `Data/Rag/` for supported file types
- Generates SHA-256 hash of file content as `document_id` (content-based deduplication)
- Extracts OS-level metadata: file size, created time, modified time
- Captures `relative_path` and `parent_folder` for each file (handles duplicate filenames across folders)
- Skips hidden files, reference datasets, and unsupported extensions
- Supported: `.pdf`, `.docx`, `.txt`, `.md`, `.json`, `.xml`, `.csv`, `.xlsx`, `.xls`, `.jpg`, `.jpeg`, `.png`

### Stage 2: Document Parser
**File:** `services/processing/parser.py`

- **PDF**: PyMuPDF per-page extraction - each page's text is preserved separately for page-aware chunking
- **DOCX**: python-docx paragraph extraction with table detection
- **Markdown (.md)**: UTF-8 text extraction, preserves formatting structure
- **CSV/XLSX/XLS**: pandas DataFrame -> string conversion with table preservation
- **TXT**: UTF-8 with Latin-1 fallback encoding
- **JSON/XML**: Structured text extraction
- **Images**: Placeholder for OCR pipeline
- Outputs: full text, per-page data (`pages_data`), total pages, word count

### Stage 3: OCR Engine
**File:** `services/processing/ocr.py`

- **Tesseract OCR** with Pillow for image preprocessing
- Conditional execution: only runs when extracted text is below `MIN_TEXT_THRESHOLD` (50 chars)
- **Image OCR**: Extracts text from JPG, JPEG, PNG, TIFF, BMP, GIF, WebP
- **Scanned PDF OCR**: Renders each page to image at 200 DPI via PyMuPDF, then OCRs each page
- Handles RGB/RGBA/grayscale conversion automatically
- Falls back gracefully if Tesseract is not installed

### Stage 4: Triage AI
**File:** `services/processing/triage.py`

The classification engine uses a **4-tier priority system**:

| Priority | Signal | Confidence | Example |
|----------|--------|-----------|---------|
| 1 (highest) | Folder path | 95% | `/shipping_orders/order_123.pdf` -> shipping_order |
| 2 | Filename keywords | 85-92% | `invoice_10256.pdf` -> invoice |
| 3 | Text content regex | 75-88% | Contains "purchase order" -> purchase_order |
| 4 | LLM auto-discovery | 80% | Unknown content -> LLM classifies and creates new type |
| Fallback | None matched | 50% | type=unknown, dept=general |

**Supported classifications (expandable):**
- **Types:** invoice, receipt, contract, agreement, report, policy, email, form, purchase_order, shipping_order, inventory_report, memo, manual, specification, proposal, resume, certificate, presentation, spreadsheet, letter, unknown - *plus any new types auto-discovered by LLM*
- **Departments:** finance, legal, hr, operations, engineering, marketing, sales, general

**Quality scoring** is deterministic and independent of classification:
- Starts at 1.0
- Penalizes low text (<50 chars): -0.3
- Penalizes noisy OCR (low alphanumeric ratio): -(1.0 - clean_ratio)
- Documents below threshold (0.60) are flagged as degraded

### Stage 5: Chunking Engine
**File:** `services/embedding/chunker.py`

- **Page-aware splitting**: For multi-page PDFs, each page is chunked separately - every chunk carries its `page_number`
- **Recursive splitting**: Tries paragraph breaks -> line breaks -> sentence breaks -> word breaks -> hard character cut
- **Overlap**: Configurable overlap (default 150 chars) between consecutive chunks for context continuity
- **Metadata inheritance**: Every chunk inherits ALL metadata from upstream stages: `document_id`, `file_name`, `folder_path`, `repository`, `document_type`, `department_category`, `quality_score`, `is_degraded`, `detected_issues`, `triage_confidence`, `page_number`, `total_pages`, etc.
- Default chunk size: 1500 characters

### Stage 6: Embedding Generator
**File:** `services/embedding/generator.py`

- Model: BAAI/bge-large-en-v1.5 (1024 dimensions)
- Batch processing: configurable batch size (default 32)
- Same model used for both ingestion and query embedding (critical for vector space alignment)
- Pre-downloaded during Docker build - no runtime downloads

### Stage 7: Vector Index
**File:** `services/search/database.py`

- **OpenSearch 2.11** with HNSW vector index
- Index config: `ef_construction=128`, `m=16`, cosine similarity via nmslib
- 18+ mapped fields: embedding, text, chunk_id, document_id, file_name, file_path, folder_path, repository, document_type, department_category, quality_score, is_degraded, detected_issues, triage_confidence, chunk_index, page_number, total_pages, word_count, file_extension
- **In-memory fallback**: If OpenSearch is unavailable, falls back to numpy cosine similarity search (development/testing)
- Bulk upload with error tracking and immediate index refresh

---

## Online Query Pipeline Deep-Dive

### Step 1: Intent Parsing
**File:** `services/search/retriever.py`

Extracts metadata filters from natural language:
- "finance invoices" -> `{document_type: "invoice", department_category: "finance"}`
- "shipping orders" -> `{document_type: "shipping_order"}`
- "in the invoices folder" -> `{folder_path: "invoices"}`
- "pdf files" -> `{file_extension: ".pdf"}`
- Handles plurals and variations ("purchase orders", "contracts", "specs")

### Step 2: Query Embedding
Same SentenceTransformer model as ingestion ensures vector space alignment between queries and documents.

### Step 3: Hybrid Search
**File:** `services/search/database.py`

- **With filters**: OpenSearch `bool` query combining `knn` vector search with `term`/`wildcard` metadata filters
- **Without filters**: Pure k-NN cosine similarity search
- Returns top 15 candidates (configurable via `TOP_K_RETRIEVAL`)

### Step 4: CrossEncoder Reranking
**File:** `services/rag/reranker.py`

- Creates `[query, chunk_text]` pairs for each candidate
- Scores all pairs with CrossEncoder model
- Returns top 5 most relevant (configurable via `TOP_K_RERANK`)
- Falls back to vector ordering on error

### Step 5: LLM Generation
**File:** `services/rag/engine.py`

Uses a **unified 3-layer context** approach:

1. **FULL DOCUMENT layer** - If a specific file is mentioned (e.g., "invoice_10256.pdf" or "tell me more about that document"), ALL chunks of that file are retrieved and included
2. **ANALYTICS layer** - Repository metadata (total counts, distributions, quality stats) for count/summary questions
3. **RETRIEVAL layer** - Semantically relevant chunks from vector search for content questions

All three layers are combined into a single LLM prompt so multi-topic questions work naturally.

**Conversation memory**: Last 4 exchanges (8 messages) are included in the LLM context for continuity.

---

## Incremental Processing & File Manifest

The system uses a persistent JSON manifest (`Data/file_manifest.json`) to enable true incremental processing:

### How It Works

1. On startup, the scanner compares the filesystem against the manifest
2. Files are classified into four categories:
   - **New**: Present on disk but not in manifest -> full pipeline processing
   - **Modified**: Content hash changed -> delete old chunks, re-process
   - **Deleted**: In manifest but not on disk -> remove chunks from OpenSearch
   - **Unchanged**: Same content hash -> skip entirely
3. After processing, each file's metadata is recorded: document_id, content_hash, document_type, department, quality_score, chunk_count, timestamps

### Performance Impact

| Scenario | Time |
|----------|------|
| First run (200 files) | ~90 seconds |
| Restart (no changes) | ~4 seconds |
| Add 5 new files | ~6 seconds (only processes 5 files) |
| Delete a file | ~1 second (removes chunks, updates manifest) |

### First-Run Migration

If OpenSearch already has data but no manifest exists (upgrading from a previous version), the system automatically builds the manifest from existing OpenSearch payloads - no re-processing required.

### Manifest Entry Fields

Each file entry tracks: `document_id`, `file_path`, `relative_path`, `parent_folder`, `file_name`, `file_extension`, `file_size_bytes`, `content_hash`, `modified_time`, `status` (processed/pending/failed), `document_type`, `department_category`, `quality_score`, `triage_confidence`, `chunk_count`, `last_processed`, `error_message`

---

## Handling Uncategorized Data at Scale

The system is designed for real-world data where documents arrive without structure or categorization:

1. **No folder structure required** - Dump all files into `Data/Rag/` and the pipeline processes everything
2. **Triage AI auto-classifies** using filename keywords and text content (Priority 2 and 3)
3. **Unclassifiable documents** get `type=unknown`, `dept=general` - they are still fully indexed and searchable via semantic search
4. **Quality scoring is independent** - Even uncategorized documents get accurate quality assessments
5. **Adding folder structure improves accuracy** - If you organize files into folders (`invoices/`, `contracts/`), triage uses Priority 1 (folder path) for 95% confidence classification
6. **Duplicate filenames handled** - Manifest is keyed by relative path, so `invoices/report.pdf` and `contracts/report.pdf` are tracked independently

---

## Persistent Chat History

Conversations are automatically saved to `Data/chat_history/conversations.json`:

- **Auto-save**: Every assistant response triggers a save
- **Survives**: Browser refresh, tab close, container restart (Data/ is a Docker volume)
- **Load previous**: Select from conversation history dropdown
- **New chat**: Start fresh while preserving old conversations
- **Retention**: Last 50 conversations kept

---

## Reliability & Fault Tolerance

| Feature | Implementation |
|---------|---------------|
| **In-memory fallback** | If OpenSearch is unavailable, search uses numpy cosine similarity - full functionality preserved |
| **Incremental processing** | File manifest ensures only changed files are re-processed, preventing wasted compute |
| **Content deduplication** | SHA-256 hash prevents duplicate document ingestion across folders |
| **Atomic manifest writes** | Write to `.tmp` then `os.replace()` prevents corruption on crash |
| **Error isolation** | Each pipeline stage has try/except - one file failure doesn't stop the batch |
| **Failed file tracking** | Manifest records failed files with error messages for debugging |
| **Rotating logs** | 5MB files, 3 backups, enterprise format with timestamps |
| **Health checks** | Docker health check on OpenSearch before app starts |
| **Graceful classification** | Unknown documents still get indexed and searched |
| **Encoding fallback** | UTF-8 -> Latin-1 for text files |
| **Chat persistence** | JSON file in Docker-mounted volume |
| **Periodic manifest save** | Saves every 25 docs during large ingestions to prevent data loss |

---

## Scalability

### Current Architecture Limits

| Dimension | Current Capacity | Bottleneck |
|-----------|-----------------|------------|
| Documents | 1,000-10,000 | Single OpenSearch node memory |
| Chunks | 100,000+ | OpenSearch HNSW index |
| Concurrent queries | 10-50 | Single Streamlit process |
| File size | Up to ~100MB each | Python memory during parsing |

### Scaling Strategies

**Vertical Scaling (Quick Wins):**
- Increase `OPENSEARCH_JAVA_OPTS` heap size (`-Xms2g -Xmx2g`) for larger datasets
- Increase `EMBEDDING_BATCH_SIZE` for faster ingestion on GPU-equipped machines
- Adjust `CHUNK_SIZE` to reduce total chunk count

**Horizontal Scaling (Production):**
- **OpenSearch cluster**: Add data nodes with index sharding (`number_of_shards: 3+`)
- **Ingestion workers**: Parallelize file processing across multiple containers
- **Load balancer**: Deploy multiple Streamlit instances behind nginx/ALB
- **Embedding service**: Separate embedding into a GPU-backed microservice (e.g., Triton Inference Server)

**Enterprise Scaling (10K+ documents):**
- Replace single-node OpenSearch with AWS OpenSearch Service (managed, auto-scaling)
- Use S3 as document storage with event-driven ingestion (S3 -> Lambda -> pipeline)
- Deploy embedding model on SageMaker endpoints for auto-scaling GPU inference
- Replace Streamlit with FastAPI backend + React frontend for higher concurrency

---

## Amazon Bedrock Migration Path

The system is designed with a clean separation between the pipeline and the LLM provider, making migration to Amazon Bedrock straightforward.

### What Changes

| Component | Current | Bedrock Equivalent |
|-----------|---------|-------------------|
| LLM Provider | Groq API (Llama 3.3 70B) | Amazon Bedrock (Claude, Llama, Titan) |
| LLM Client | `groq` Python SDK | `boto3` Bedrock Runtime |
| Embedding Model | Local SentenceTransformer | Amazon Titan Embeddings or keep local |
| Vector Database | Self-hosted OpenSearch | Amazon OpenSearch Service (managed) |
| Document Storage | Local filesystem | Amazon S3 |
| OCR | Local Tesseract | Amazon Textract |
| Container | Docker Compose | Amazon ECS / EKS |

### Migration Steps

**1. LLM - Replace Groq with Bedrock (1 file change)**

Only `services/rag/engine.py` calls the LLM. Replace the Groq client:

```python
# Current (Groq)
from groq import Groq
client = Groq(api_key=settings.GROQ_API_KEY)
response = client.chat.completions.create(model=settings.GROQ_MODEL_NAME, ...)

# Bedrock equivalent
import boto3
client = boto3.client("bedrock-runtime", region_name="us-east-1")
response = client.invoke_model(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    body=json.dumps({"messages": [...], "max_tokens": 4096}),
)
```

**2. Vector Database - Managed OpenSearch**

Replace `OPENSEARCH_HOST=opensearch` with your Amazon OpenSearch Service endpoint. The `opensearch-py` client works identically with managed OpenSearch - only the connection config changes.

**3. Document Storage - S3 Integration**

Replace `LocalSystemConnector` in `services/ingestion/connectors.py` with an S3 connector:
- Use `boto3.client('s3')` to list and download objects
- S3 event notifications trigger ingestion for new/modified/deleted files
- FileManifest maps S3 keys instead of local paths

**4. OCR - Amazon Textract**

Replace `services/processing/ocr.py` Tesseract calls with Textract API:
- `boto3.client('textract').detect_document_text()` for simple OCR
- `start_document_analysis()` for tables and forms extraction
- Higher accuracy than Tesseract, no infrastructure to manage

**5. Deployment - ECS/EKS**

- Push Docker image to Amazon ECR
- Deploy as ECS Fargate task (serverless containers)
- Use Application Load Balancer for HTTPS and scaling
- Store environment variables in AWS Systems Manager Parameter Store

### Cost Estimate (Bedrock)

| Component | Approximate Cost |
|-----------|-----------------|
| Bedrock Claude Sonnet (per 1M tokens) | ~$3 input / ~$15 output |
| OpenSearch Service (single-node) | ~$70-150/month |
| ECS Fargate (1 vCPU, 4GB) | ~$35/month |
| S3 storage (10GB) | ~$0.25/month |
| Textract (1000 pages/month) | ~$1.50/month |

---

## Configuration Reference

All settings are in `.env` and managed through `core/config.py` (Pydantic BaseSettings):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | Your Groq API key (**required**) |
| `GROQ_MODEL_NAME` | `llama-3.3-70b-versatile` | LLM model for answer generation |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-large-en-v1.5` | Embedding model (auto-downloads) |
| `EMBEDDING_DIMENSIONS` | `1024` | Vector dimensions |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks per embedding batch |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `OPENSEARCH_HOST` | `localhost` | OpenSearch host (auto-set to `opensearch` in Docker) |
| `OPENSEARCH_PORT` | `9200` | OpenSearch port |
| `VECTOR_INDEX_NAME` | `enterprise_docs` | OpenSearch index name |
| `CHUNK_SIZE` | `1500` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `15` | Initial retrieval candidates |
| `TOP_K_RERANK` | `5` | Final chunks after reranking |
| `MIN_TEXT_THRESHOLD` | `50` | Minimum chars before triggering OCR |
| `TRIAGE_QUALITY_THRESHOLD` | `0.60` | Below this = degraded document |
| `DATA_DIR` | `Data/Rag` | Document repository path |
| `MANIFEST_FILE` | `Data/file_manifest.json` | File state tracking manifest |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ENVIRONMENT` | `production` | Environment mode |

All values are parameterized - change any setting in `.env`, rebuild, and the entire pipeline + dashboard reflects the change.

---

## Project Structure

```
Enterprise-Content-Intelligence/
├── core/                              # Foundation layer
│   ├── config.py                      # Centralized settings (Pydantic, from .env)
│   ├── models.py                      # Data models (DocumentMetadata, Chunk, RAGResponse)
│   └── logger.py                      # Rotating file + console logging
├── services/
│   ├── ingestion/
│   │   └── connectors.py              # LocalSystemConnector + FileManifest
│   ├── processing/
│   │   ├── parser.py                  # Multi-format parser (PDF, DOCX, MD, CSV, etc.)
│   │   ├── ocr.py                     # Conditional OCR for low-text documents
│   │   └── triage.py                  # 4-tier classification + quality scoring
│   ├── embedding/
│   │   ├── chunker.py                 # Page-aware recursive chunking with metadata
│   │   └── generator.py               # SentenceTransformer batch embedding
│   ├── search/
│   │   ├── database.py                # OpenSearch HNSW client + in-memory fallback
│   │   └── retriever.py               # Intent parsing + folder/ext filters + hybrid search
│   ├── rag/
│   │   ├── engine.py                  # 3-layer unified context RAG orchestrator
│   │   └── reranker.py                # CrossEncoder reranking
│   ├── analytics/
│   │   └── intelligence.py            # Repository metrics + degradation reports
│   └── chat/
│       └── history.py                 # Persistent JSON chat history
├── ui/
│   └── app.py                         # Streamlit 4-page dashboard
├── scripts/
│   └── ingest.py                      # Manifest-based incremental ingestion pipeline
├── Data/
│   ├── Rag/                           # Document repository (any structure)
│   ├── file_manifest.json             # Persistent file state tracker
│   └── chat_history/                  # Persistent conversation storage
├── architecture.mmd                   # Mermaid pipeline diagram (full visual)
├── docker-compose.yml                 # OpenSearch 2.11 + App container
├── Dockerfile                         # Python 3.11 + Tesseract + CPU-only PyTorch + ML models
├── requirements.txt                   # All dependencies
├── .env                               # Configuration (not committed)
└── .env.example                       # Configuration template
```

---

## Stopping & Restarting

```bash
# Stop (data persists in OpenSearch volume + manifest + chat history)
docker-compose down

# Stop and remove ALL data (fresh start)
docker-compose down -v

# Restart (data persists, models cached, manifest skips unchanged files)
docker-compose up

# Rebuild after code changes
docker-compose up --build
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OpenSearch won't start | Increase Docker memory to 4GB+ in Docker Desktop -> Settings -> Resources |
| Slow first startup | Normal - downloads ~1.3GB of ML models during Docker build. Cached after first build |
| "Groq API key not configured" | Check `.env` file has a valid `GROQ_API_KEY` |
| Port 8501 in use | Change port in `docker-compose.yml`: `"8502:8501"` |
| Port 9200 in use | Change OpenSearch port or stop conflicting service |
| Dashboard loads slowly on first visit | Normal - ML models loading into RAM (~10-20s). Instant after first load |
| Chat history lost | Ensure `Data/` directory is volume-mounted in `docker-compose.yml` |
| Documents not classified correctly | Organize into named folders for 95% confidence, or check filename matches triage keywords |
| Search returns no results | Verify documents were ingested (check Pipeline Monitor page) |
| "OpenSearch unavailable" warning | System auto-falls back to in-memory mode - functional but not persistent |
| Files not re-processed after edit | Delete `Data/file_manifest.json` to force full re-ingestion |
| PyTorch download slow on Windows | CPU-only PyTorch is pre-installed in Dockerfile (~200MB vs ~915MB GPU version) |

---

Built with Python, OpenSearch, and open-source ML models. Designed for enterprise-grade document intelligence.
