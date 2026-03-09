# Vector Database Feature Comparison (2025–2026)

Compiled: March 2026. Data sourced from official documentation, release notes, and vendor announcements.

Products covered:
1. Pinecone
2. Weaviate
3. Milvus / Zilliz Cloud
4. Qdrant
5. Chroma
6. DataStax Astra DB

---

## 1. Pinecone

**Type:** Fully managed, proprietary cloud service
**Website:** https://www.pinecone.io

### Indexing Algorithms
- HNSW (Hierarchical Navigable Small World) — primary ANN algorithm for dense indexes
- IVF (Inverted File Index) combined with PQ (Product Quantization) for large-scale compression
- Log-structured merge tree approach that dynamically balances indexing strategy by workload
- Scalar quantization for agentic/small-slab workloads
- Partition-based indexing for large datasets
- Dense indexes for semantic/vector search
- Sparse indexes for lexical/keyword search (sparse vectors, e.g., BM25-style)

### Distance Metrics
- Cosine similarity
- Euclidean distance (returns squared Euclidean distance as score)
- Dot product
- Manhattan distance is NOT supported

### Filtering / Metadata Support
- Metadata filtering on all queries (e.g., `{"category": {"$eq": "technology"}}`)
- Namespace-scoped filtering (all operations target one namespace)
- Hybrid search: combines vector similarity with metadata filtering
- Metadata schema management: pre-declare which fields should be indexed for filtering (2025 feature)
- Reranking integrated into search pipeline

### Storage
- Cloud-only (no on-premises option)
- Serverless architecture: BLOB/object storage as source of truth, compute separated from storage
- Pod architecture: pre-configured hardware units on AWS, GCP, or Azure
- Dedicated read nodes (early access, 2025) for high-QPS workloads

### Persistence Model
- Fully managed — persistence handled transparently by the platform
- Data is durably stored in object storage; indexes rebuilt from storage on demand
- No user-managed WAL or snapshot configuration

### Client-Server Architecture
- Pure SaaS; no self-hosted option
- REST API and gRPC
- Two deployment models: Serverless (auto-scaling) and Pod-based (fixed hardware)
- Bring Your Own Cloud (BYOC) available on GCP (2025) for high-compliance deployments

### Embedding Model Integrations
- Integrated embedding: specify a hosted model (e.g., `multilingual-e5-large`), upsert raw text, Pinecone vectorizes automatically
- External embeddings: upsert pre-computed vectors directly
- Third-party providers: OpenAI, Cohere, and others
- Integrated reranking supported (SDK v4.0.0+)

### Partial / Lazy Loading
- Not applicable in the traditional sense — serverless architecture loads data on demand from object storage
- Import and incremental upsert support for partial data loading workflows

### Sharding / Clustering
- Serverless: automatic, transparent sharding and scaling
- Pod architecture: horizontal scaling by adding pods; vertical scaling by upgrading pod type
- Data partitioned across pods for parallel processing

### Auth / Security
- API key authentication
- Role-Based Access Control (RBAC)
- Single Sign-On (SSO)
- End-to-end encryption (in transit and at rest)
- SOC 2 Type II and GDPR compliant
- BYOC on GCP for organizations requiring data residency

### Payload / Metadata Storage
- Metadata stored alongside vectors in the index
- Each vector record can carry arbitrary key-value metadata
- Metadata fields can be pre-declared in a schema for optimized filtering (2025)

### Multi-Tenancy
- Namespace-based isolation: each namespace is a logical partition within an index
- One namespace per customer is the recommended multi-tenancy pattern
- All queries, upserts, and deletes target a single namespace

### SDK Languages
- Python
- Node.js (JavaScript/TypeScript)
- Java
- Go

### Unique Differentiating Features
- Only major vector database that is serverless-first with no self-hosted option
- Automatic scaling with zero infrastructure management
- Hybrid dense + sparse index architecture in one platform (semantic + lexical)
- Dedicated read nodes for predictable latency at high QPS (2025)
- Purpose-built Rust engine underlying the serverless infrastructure

---

## 2. Weaviate

**Type:** Open-source + managed cloud (Weaviate Cloud Services)
**Website:** https://weaviate.io
**Language:** Written in Go

### Indexing Algorithms
- **HNSW** — primary vector index; custom full-CRUD implementation with Write-Ahead Log; logarithmic time complexity
- **Flat Index** — disk-backed, single-layer; small memory footprint; best for small/multi-tenant collections; linear time complexity
- **Dynamic Index** — starts as flat, auto-upgrades to HNSW when object count exceeds threshold (default: 10,000); one-way conversion
- **HFresh** — cluster-based index using SPFresh algorithm; uses HNSW for centroid search; balances memory efficiency vs. search performance for large datasets
- **Inverted Indexes** — support BM25 full-text search and accelerate filtering
- HNSW snapshots to reduce startup time for large indexes

### Distance Metrics
- Cosine (default)
- L2 / Euclidean (l2-squared)
- Dot product
- Hamming
- Manhattan
- Note: HFresh index supports only cosine and l2-squared

### Filtering / Metadata Support
- Pre-filter before vector search or post-filter after
- Rich structured filtering on object properties
- AutoCut: automatically limits results based on discontinuities in distance/score
- Score threshold filtering (maximum cosine distance)
- Per-collection, per-vector-space filter configuration
- BM25F weighting per field in keyword search
- Hybrid search fusing BM25 + vector with configurable fusion (Reciprocal Rank Fusion or Relative Score Fusion)

### Storage
- **On-premises / self-hosted**: Docker or Kubernetes, data persisted to mounted volumes
- **Weaviate Cloud (WCS)**: fully managed, persistent by default
- **Hybrid SaaS**: cluster runs in customer's cloud, managed remotely by Weaviate
- Object and inverted index data stored using LSM-Tree segmentation
- Vector index stored with WAL and optional snapshots
- Bloom filters for efficient segment navigation without full merging

### Persistence Model
- WAL (Write-Ahead Log) for disaster recovery and HNSW state persistence
- HNSW snapshots for fast startup of large indexes
- Inverted index uses LSM-Tree with Bloom filter-assisted segment traversal
- Persistent across restarts when volume is mounted (Docker/Kubernetes)
- Object TTL (time-to-live) for automatic data expiry (technical preview, v1.35)

### Client-Server Architecture
- Client-server: Weaviate server exposes REST, GraphQL, and gRPC APIs
- Official client SDKs communicate with the server
- Embedded mode (in-process) available for local development
- Deployment options: Docker, Kubernetes, Weaviate Cloud, Hybrid SaaS

### Embedding Model Integrations
- Built-in vectorizer modules: `text2vec-openai`, `text2vec-cohere`, `text2vec-huggingface`, `text2vec-palm` (Google), `text2vec-transformers` (self-hosted BERT-family)
- Image embeddings: `img2vec-neural`
- Q&A module: `qna-transformers` for direct question answering
- Generative modules: `generative-openai`, `generative-cohere`, etc. (RAG directly in DB)
- Reranker modules: `reranker-cohere`, `reranker-transformers`
- Weaviate Embeddings (cloud-native embedding service, 2025) — only on WCS, not self-hosted
- Bring your own vectors (no vectorizer required)

### Partial / Lazy Loading
- Inactive multi-tenant shards can be moved to cold storage and reactivated on demand
- Hot/warm/cold tiering for tenant data
- HNSW snapshots enable fast partial reload at startup

### Sharding / Clustering
- Horizontal scaling: each collection can span multiple shards across nodes
- Each tenant in multi-tenancy gets a dedicated shard
- Replication for fault tolerance and read throughput
- Kubernetes-native deployment for cluster management

### Auth / Security
- API key authentication
- Username + password authentication
- OIDC (OpenID Connect) authentication — runtime-configurable since v1.35 (no restart needed)
- RBAC (Role-Based Access Control) — fine-grained permissions
- HIPAA compliance (Weaviate Cloud, 2025)
- Social login for Weaviate Cloud
- Mutual TLS support

### Payload / Metadata Storage
- Objects stored with arbitrary properties (schema-defined or schema-free)
- Properties indexed via inverted index for filtering
- Each indexed property gets a dedicated inverted index bucket per tenant shard
- Objects and vectors stored together, allowing object retrieval alongside search results

### Multi-Tenancy
- First-class multi-tenancy: each tenant gets a dedicated shard within a collection
- Tenant-level isolation: dedicated HNSW index, dedicated inverted index, dedicated storage
- Hot/cold/warm storage tiering per tenant to reduce resource consumption for inactive tenants
- Tenant deletion does not affect other tenants
- Scales to millions of tenants per cluster

### SDK Languages
- Python (v4 client)
- TypeScript / JavaScript
- Go
- Java (community/official)
- REST and GraphQL accessible from any HTTP client

### Unique Differentiating Features
- **GraphQL-first query interface** with rich nesting, filtering, and metadata in a single call
- **Generative search (RAG) built directly into the database** — results can be passed to an LLM module without leaving Weaviate
- **Module system**: plug-in architecture for vectorization, inference, reranking, and enrichment
- **Query Agent** (GA 2025): AI-powered agent that performs query expansion, decomposition, schema introspection, and reranking automatically
- **Hybrid search with configurable fusion algorithms** (RRF and relativeScoreFusion)
- **Object TTL** for automatic stale-data expiry

---

## 3. Milvus / Zilliz Cloud

**Type:** Open-source (Milvus) + managed cloud (Zilliz Cloud)
**Website:** https://milvus.io / https://zilliz.com
**License:** Apache 2.0 (Milvus); under LF AI & Data Foundation
**Language:** C++ core, Go coordination layer

### Indexing Algorithms
- **FLAT** — brute-force exact search; baseline accuracy
- **IVF_FLAT** — inverted file index; clusters vectors, searches nearest clusters; no compression
- **IVF_SQ8** — IVF + scalar quantization (FLOAT to UINT8); significant memory savings
- **IVF_PQ** — IVF + product quantization; further compression for large datasets
- **HNSW** — graph-based; best for high search-efficiency demands; multiple HNSW variants: HNSW_SQ, HNSW_PQ, HNSW_PRQ
- **SCANN** — Google's ScaNN algorithm for high-recall, fast search
- **DiskANN** — disk-based ANN index for datasets exceeding memory
- **GPU indexes** — CAGRA (NVIDIA), IVF_FLAT_GPU, IVF_PQ_GPU for GPU-accelerated search
- **Sparse vector indexes** — for BM25, SPLADE, BGE-M3 (hybrid retrieval)
- AutoIndex on Zilliz Cloud (AI-powered index selection)

### Distance Metrics
- L2 / Euclidean distance
- Inner Product (IP) / Dot product
- Cosine similarity
- Hamming (binary vectors)
- Jaccard (binary vectors)

### Filtering / Metadata Support
- Metadata filtering on scalar fields during search (pre- and post-filtering)
- Range search (find vectors within a distance radius)
- Partition key-based filtering for multi-tenancy
- BM25 full-text search natively supported
- Hybrid search: dense + sparse vectors in same collection, reranking across result sets
- Time Travel: query data as of a specific timestamp

### Storage
- Cloud-native separated storage and compute
- **In-memory**: data loaded into query node memory for search
- **Disk-based (DiskANN)**: large datasets stored on NVMe SSD
- **Memory-mapped (mmap)**: compressed vectors in mmap for cost-effective large-scale search
- **Hot/cold tiering** (Milvus 2.6+): frequently accessed data in memory/SSD, cold data in slower storage
- Object storage backend (S3-compatible) as persistent storage layer
- Milvus Lite: single-process embedded mode for local dev (no server required)
- Milvus Standalone: single-node server
- Milvus Distributed: Kubernetes-native, multi-node

### Persistence Model
- Object storage (S3/MinIO/Azure Blob/GCS) as durable source of truth
- WAL (Write-Ahead Log) for crash recovery
- Log broker (Pulsar or Kafka) for streaming ingestion and replication
- Snapshots and time-travel backups
- Separate storage nodes from compute nodes; stateless query nodes reload from object storage

### Client-Server Architecture
- Client-server: gRPC + RESTful APIs
- Three deployment tiers:
  - **Milvus Lite**: embedded Python library (no server)
  - **Milvus Standalone**: single Docker container
  - **Milvus Distributed**: Kubernetes cluster with separate coordinator, query, data, and index nodes
- Zilliz Cloud: fully managed Milvus with Serverless, Dedicated, and BYOC options

### Embedding Model Integrations
- PyMilvus integrates with: OpenAI, Cohere, HuggingFace, Sentence Transformers, SPLADE, BGE-M3, and others
- Native BM25 support for sparse vector generation
- LangChain and LlamaIndex integrations (Milvus as vector store)
- Vector Transmission Services (VTS) for data migration from Elasticsearch, pgvector, and other Milvus instances

### Partial / Lazy Loading
- **Lazy loading** (Milvus 2.6+): at initialization, only segment-level metadata (schema, index info, chunk mappings) is cached — no field data or index files loaded upfront
- **Partial loading** (Milvus 2.6+): only requested fields loaded during query; other scalar fields fetched lazily on access
- **Tenant-aware loading**: for multi-tenant workloads, only the vector index partition for the queried tenant is loaded
- **LRU-based cache eviction** when local capacity is exceeded

### Sharding / Clustering
- Horizontal sharding: data partitioned across query nodes and storage shards
- Collections can have multiple partitions (logical sub-divisions)
- Partition key support for tenant-level data isolation
- Replication factor configurable per collection
- Kubernetes-native: separate scaling of coordinator, query, data, and index node pools
- Zero-downtime rolling updates

### Auth / Security
- User authentication (mandatory when enabled)
- TLS encryption for all inter-node and client communications
- RBAC (Role-Based Access Control) with fine-grained permissions per collection/database
- Multi-level isolation: database, collection, partition, or partition key
- Zilliz Cloud: additional enterprise security features, private networking

### Payload / Metadata Storage
- Scalar fields stored alongside vectors in the same collection
- Supports rich data types: string, int, float, bool, JSON, array
- Dynamic schema support: add fields without recreating collection
- Scalar fields indexed separately (bitmap, inverted, trie indexes) for filtering performance

### Multi-Tenancy
- Strategies: one database per tenant, one collection per tenant, one partition per tenant, or partition key per tenant
- Partition key: hash-based routing to logical partitions within a collection (most scalable for millions of tenants)
- Tenant-aware lazy loading (Milvus 2.6+) ensures only active tenant data is loaded
- Fine-grained RBAC access control per tenant

### SDK Languages
- Python (PyMilvus — primary, richest feature set)
- Java
- Go
- Node.js (JavaScript/TypeScript)
- RESTful API (language-agnostic)

### Unique Differentiating Features
- **Widest index selection** of any vector database: FLAT, IVF variants, HNSW variants, SCANN, DiskANN, GPU indexes
- **GPU-accelerated indexing** with CAGRA (NVIDIA) — up to 10x faster index builds
- **Lazy and partial loading** (Milvus 2.6+) — industry-leading memory efficiency for large datasets
- **Time Travel**: query historical snapshots of data at any timestamp
- **Vector Transmission Services (VTS)**: built-in tools to migrate vectors from other databases
- **Milvus Lite**: zero-infrastructure embedded mode runnable as a Python library
- **Trillion-vector scale** design — largest-scale open-source vector database

---

## 4. Qdrant

**Type:** Open-source + managed cloud (Qdrant Cloud)
**Website:** https://qdrant.tech
**Language:** Written in Rust

### Indexing Algorithms
- **HNSW** — sole dense vector index type; custom Rust implementation with incremental building on segment merges (v1.14+), parallel batch search, and memory optimizations
- **Sparse vector index** — for sparse vectors (BM25/TF-IDF generalization, SPLADE)
- **Payload indexes** — per-field indexes (similar to document DB secondary indexes); support: keyword, integer, float, geo, datetime, full-text, UUID
- **Mutable Map Index** — boosted performance for full-text, integer, and other indexes (v1.15+)
- **Inline storage** (v1.16+) — quantized vectors stored directly inside HNSW nodes for single-disk-read performance in low-RAM setups

### Distance Metrics
- Cosine similarity
- Dot product
- Euclidean (L2)
- Manhattan
- Hamming (for binary vectors)

### Filtering / Metadata Support
- Payload-based filtering with rich condition types: `must`, `should`, `must_not` (boolean logic)
- Supported field types for filtering: keyword, integer, float, datetime, geo (bounding box, radius, polygon), full-text, UUID
- Nested JSON payload conditions
- **ACORN mechanism**: improves accuracy for high-cardinality, multi-condition filters
- Query planner uses payload index statistics to choose optimal search strategy (filter-first vs. vector-first)
- **Strict mode**: prevents unindexed field queries, caps result sizes, limits filter complexity, rate-limits read/write
- Filterable index option for explicit filter-optimized HNSW variants

### Storage
- **In-memory**: default for small collections and active data
- **On-disk (RocksDB)**: `OnDisk` payload storage for large payloads; vectors can be mmap'd to disk
- **Inline storage** (v1.16+): HNSW + quantized vectors on disk in co-located pages; enables order-of-magnitude better disk search performance
- **io_uring** for async disk I/O maximizing NVMe/network-attached storage throughput
- Quantization options: scalar, product, binary, 1.5-bit, 2-bit, asymmetric (v1.16+ additions)

### Persistence Model
- Write-Ahead Log (WAL) for all updates — confirms data persistence before acknowledging writes, survives power outages
- Segment-based storage: each collection divided into independent segments with their own vector and payload storage
- Snapshot and restore support
- RocksDB for on-disk payload and metadata

### Client-Server Architecture
- Client-server: Qdrant server exposes REST (HTTP) and gRPC APIs
- Official clients communicate over these APIs
- Self-hostable via Docker or Kubernetes
- Qdrant Cloud: fully managed with free tier
- Consistent API across cloud, hybrid, and edge deployments

### Embedding Model Integrations
- Does not provide built-in embedding inference; users bring their own vectors
- 35+ integrations (as of 2025), including: LangChain, LlamaIndex, OpenAI, Cohere, HuggingFace, Fastembed (Qdrant's own lightweight embedding library)
- **Fastembed**: Qdrant's own Python library for CPU-optimized local embedding generation (ONNX-based)
- Native support for dense, sparse, and multi-vector embeddings
- Official n8n node for no-code workflows

### Partial / Lazy Loading
- Segment-level architecture: segments loaded independently
- mmap option for vectors: accessed from disk without full load into RAM
- On-disk payload: large payloads read from RocksDB on demand
- Indexed payload fields kept in RAM regardless of payload storage mode

### Sharding / Clustering
- Collection-level sharding: each collection split into N shards (recommended: 12 shards for flexible node count scaling)
- Replication factor configurable; production recommendation: factor >= 2
- Dynamic scaling: zero-downtime rolling updates, seamless shard addition
- Tenant-aware sharding: sub-indexes per tenant within a shared collection for efficient multi-tenant search
- Tiered multitenancy (v1.16+): small tenants share shards, large tenants can be promoted to dedicated shards
- Improved shard recovery and resharding resilience (v1.14+)

### Auth / Security
- API key authentication
- JWT (JSON Web Token) based granular access control
- RBAC built on top of JWT: read-only, read-write, admin roles per collection
- Qdrant Cloud: SSO (Single Sign-On), cloud RBAC, Terraform-enabled Cloud API
- TLS for in-transit encryption
- Protection against embedding inversion attacks is a stated security motivation

### Payload / Metadata Storage
- Payloads are arbitrary JSON attached to each point (vector record)
- Any JSON-serializable data supported
- `OnDisk` storage mode for large payloads (RocksDB-backed)
- Indexed payload fields kept in RAM for fast filtering regardless of storage mode
- Payload indexes support geospatial, full-text, numeric, datetime types

### Multi-Tenancy
- Tenant isolation via payload field (e.g., `tenant_id`) with payload index
- Tenant-aware HNSW sub-indexes: disables global index, builds per-tenant sub-index
- Tiered multitenancy (v1.16+): combines small and large tenants in one collection; large tenants auto-promoted to dedicated shards
- RBAC enforces data isolation between tenants

### SDK Languages
- Python
- JavaScript / TypeScript
- Rust
- Go
- .NET (C#)
- Java
- HTTP and gRPC APIs for any language

### Unique Differentiating Features
- **Rust implementation** — lowest-latency, highest-reliability foundation; SIMD (AVX512/Neon) acceleration
- **Richest payload filtering** of any vector DB: nested JSON, geo, full-text, datetime with ACORN accuracy boost for complex filters
- **Inline storage** (v1.16+): disk-based vector search with accuracy comparable to in-memory, using a single disk read per HNSW traversal
- **Tiered multitenancy** (v1.16+): first-class support for mixed small/large tenant workloads without dedicated clusters
- **Fastembed**: Qdrant's own lightweight CPU embedding library (no GPU or network required)
- **GPU-accelerated HNSW indexing** (v1.16+): up to 10x faster index builds
- **Strict mode**: production safety guardrails preventing misconfigured queries from degrading cluster performance

---

## 5. Chroma

**Type:** Open-source + managed cloud (Chroma Cloud)
**Website:** https://www.trychroma.com
**Language:** Python + Rust core (rewritten in Rust in 2025)

### Indexing Algorithms
- **HNSW** — primary vector index; logarithmic time complexity
- **IVF (Inverted File Index)** — also available for similarity search acceleration
- **Inverted index** — for full-text/BM25 search
- Custom binary format for vector storage alongside SQLite for metadata

### Distance Metrics
- Cosine similarity
- Euclidean (L2) distance
- Inner product / dot product (configurable per collection)

### Filtering / Metadata Support
- Metadata filtering on all queries: equality, inequality, numeric ranges, string matching
- `where` clause for metadata conditions
- `where_document` clause for full-text / content filtering
- Regex search support
- BM25 and SPLADE sparse vector support (first-class, 2025)
- Top-K result filtering after vector search

### Storage
- **EphemeralClient**: in-memory only, no persistence (testing/prototyping)
- **PersistentClient**: local disk storage (SQLite + custom vector binary format)
- **HttpClient**: connects to remote Chroma server
- **Chroma Cloud**: serverless object storage backend (cloud-agnostic: AWS, GCP, Azure)
- Object storage with query-aware data tiering and caching (Chroma Cloud)
- Copy-on-write for fast collection duplication

### Persistence Model
- Persistent mode writes SQLite (metadata) + custom binary (vectors) to a local directory
- Single directory copy = full database migration
- Chroma Cloud: managed persistence on object storage; no user-managed WAL
- push-based, morsel-driven Rust execution engine handles durability in cloud mode

### Client-Server Architecture
- **Embedded mode** (`EphemeralClient`, `PersistentClient`): in-process, no network, ideal for local dev
- **Client-server mode** (`HttpClient`): HTTP API to a running Chroma server; enables multi-client access
- Chroma Cloud: serverless, multi-tenant SaaS
- Server exposes HTTP REST API

### Embedding Model Integrations
- Default: Sentence Transformers (all-MiniLM-L6-v2 or similar) — runs locally without any API key
- OpenAI embeddings
- Cohere embeddings (multilingual)
- Google (Vertex AI / Gemini)
- HuggingFace
- OpenCLIP (multimodal: image + text)
- Custom embedding functions supported
- LangChain and LlamaIndex as primary orchestration frameworks

### Partial / Lazy Loading
- Serverless Chroma Cloud uses query-aware data tiering: cold data stays in object storage until needed
- Morsel-driven execution: processes data in small chunks rather than full dataset loads
- No explicit lazy loading API exposed to users

### Sharding / Clustering
- Chroma Cloud: distributed architecture with query nodes + compactor nodes separated
- Dynamic sharding with automated rebalancing (Chroma Cloud)
- Memory pressure metrics and query distribution patterns trigger shard rebalancing
- Self-hosted Chroma: single-node only (no built-in clustering for self-hosted)

### Auth / Security
- **Chroma v1.0.x does NOT include native authentication** — external reverse proxy / security layer required for self-hosted
- Chroma Cloud: managed security with standard cloud provider controls (AWS/GCP/Azure)
- Community cookbook covers securing Chroma 1.0.x via external means
- No built-in RBAC in open-source version (as of 2025)

### Payload / Metadata Storage
- Metadata stored in SQLite alongside vector store
- Arbitrary key-value JSON metadata per document
- Document content (raw text) stored and searchable

### Multi-Tenancy
- Logical hierarchy: Tenant → Database → Collection
- Tenants represent organizations; databases represent applications; collections group documents
- Chroma Cloud handles billions of records across multi-tenant environments
- Self-hosted multi-tenancy is logical only (no physical isolation between tenants)

### SDK Languages
- Python (primary, richest feature set)
- JavaScript / TypeScript
- REST API accessible from any language

### Unique Differentiating Features
- **Simplest developer experience** — start with `pip install chromadb` and three lines of code; no infrastructure required
- **Default local embedding** — Sentence Transformers run in-process; zero API keys needed for prototyping
- **Persistent-to-portable**: entire database in one directory, trivially copied or moved
- **2025 Rust-core rewrite**: 4x faster writes and queries; eliminates Python GIL bottlenecks; multithreaded parallel embedding processing
- **Serverless cloud architecture** with object storage: query nodes and compactor nodes fully separated
- **Multimodal search via OpenCLIP**: image and text in the same collection
- Lowest barrier to entry among all options — ideal for RAG prototyping and small to medium deployments

---

## 6. DataStax Astra DB

**Type:** Fully managed cloud service (built on Apache Cassandra)
**Website:** https://www.datastax.com
**Note:** IBM acquired DataStax in May 2025

### Indexing Algorithms
- **SAI (Storage-Attached Indexing)** — DataStax's proprietary index technology deeply integrated with Cassandra's storage engine (Memtable + SSTable)
- SAI uses **HNSW** via the **JVector** library for ANN vector search
- Indexes both in-memory (Memtable) and on-disk (SSTable) data as data is written
- JVector: Java-native ANN library; production-tested in Astra DB
- Secondary indexes on any column (non-vector fields) using SAI
- No support for exact KNN — ANN only (by design for performance at scale)

### Distance Metrics
- Cosine similarity
- Dot product
- Euclidean (L2) distance

### Filtering / Metadata Support
- SAI multi-predicate filtering: create SAI indexes on any column, combine multiple filters
- ANN + filter: SAI loads a superset based on predicates, sorts by vector similarity, returns top-K
- Native Cassandra Query Language (CQL) for filter expressions
- Up to 10 SAI indexes per table by default (configurable)
- JSON and structured data types supported via Cassandra's type system

### Storage
- Fully cloud-based: AWS, Azure, GCP
- Serverless (pay-as-you-go) and Dedicated cluster options
- Built on Apache Cassandra SSTable storage (Log-Structured Merge Tree)
- SAI indexes streamed with SSTables via Zero-Copy Streaming (ZCS) — no index rebuild on node bootstrap/decommission
- Disk-persistent; no pure in-memory mode
- Petabyte-scale storage capacity by design

### Persistence Model
- Cassandra's LSM-tree (Log-Structured Merge Tree) storage with commit log (WAL equivalent)
- SSTable-based storage: data written immutably, compacted in background
- SAI indexes co-located with SSTables; streamed atomically during node operations
- Backups and point-in-time restore via Astra tooling
- Fully managed; no user-managed compaction or snapshot configuration

### Client-Server Architecture
- Fully managed SaaS — no self-hosted Astra DB option
- APIs: **Astra Data API** (simplified REST, JSON-based), **CQL** (native Cassandra protocol), DevOps API, Astra CLI
- Private endpoints: AWS PrivateLink, Azure Private Link, GCP Private Service Connect
- Langflow: visual, low-code RAG application builder integrated with Astra DB

### Embedding Model Integrations
- **Astra Vectorize**: server-side embedding generation — upsert raw text, Astra generates embeddings automatically
  - Supported providers: OpenAI, Azure OpenAI, NVIDIA NeMo / NeMo Retriever, Cohere, HuggingFace, Google Vertex AI
  - Embeddings > 800/sec per GPU with NVIDIA NeMo; 4,000+ TPS ingestion
  - Single-digit millisecond latency embedding + indexing on NVIDIA H100
- Bring your own vectors: pre-computed embeddings upserted directly via Data API or CQL
- LangChain and LlamaIndex integrations
- watsonx (IBM) integration post-acquisition

### Partial / Lazy Loading
- Not applicable in the traditional sense
- SAI lazily filters: loads a superset of candidates, evaluates predicates, then ranks by vector similarity
- Cold-start behavior managed transparently by managed service

### Sharding / Clustering
- Cassandra's native consistent hashing: data distributed across nodes via virtual nodes (vnodes)
- Multi-region, multi-cloud replication built-in
- Zero-copy streaming (ZCS): SAI indexes travel with SSTables during node joins/leaves — no index rebuild
- Linear horizontal scalability (Cassandra design)
- Replication factor configurable per keyspace

### Auth / Security
- Token-based authentication (Astra application tokens)
- SSO via SAML 2.0: Entra ID, Okta, OneLogin, Google Identity Platform, Ping Identity
- Private networking: AWS PrivateLink, Azure Private Link, GCP Private Service Connect
- Compliance: PCI DSS, SOC 2, HIPAA, GDPR
- Azure OpenAI integration inherits Azure private networking + content filtering
- IBM watsonx integration adds governance and access control layers

### Payload / Metadata Storage
- Vectors stored as native `VECTOR<FLOAT, N>` type in Cassandra tables
- All other columns are standard Cassandra data types (text, int, UUID, timestamp, map, list, set, etc.)
- Rich data model: Cassandra's wide-column store alongside vector fields
- Multi-model support: tabular, vector, and (via Langflow integration) graph data

### Multi-Tenancy
- Logical multi-tenancy via Cassandra keyspaces and tables
- Role-based access control with scoped credentials per database
- One-to-one relationship between embedding provider credentials and collections (for Vectorize)
- Enterprise multi-tenant reference architecture via Langflow + Astra

### SDK Languages
- Python
- TypeScript / JavaScript
- Java
- CQL drivers available for many languages (Python, Java, Go, .NET, C++, Node.js via DataStax drivers)

### Unique Differentiating Features
- **Only vector database built natively on Apache Cassandra** — inherits proven multi-region, multi-cloud, limitless-scale architecture
- **Zero-Copy Streaming (ZCS)**: SAI indexes streamed atomically with SSTables — no index rebuild on cluster topology changes; unique to Cassandra architecture
- **Real-time vector updates**: vector changes available to queries immediately (no async index rebuild lag)
- **True multi-model**: tabular + vector + streaming data in one Cassandra table
- **Astra Vectorize**: server-side embedding generation with NVIDIA NeMo achieving production-grade throughput
- **PCI + HIPAA + SOC2** compliance from day one — enterprise-grade compliance posture
- **Trillion-vector scalability** via Cassandra's horizontal architecture (no practical upper bound stated by vendor)
- IBM acquisition (2025): watsonx governance integration and enterprise AI roadmap alignment

---

## Summary Comparison Table

| Feature | Pinecone | Weaviate | Milvus / Zilliz | Qdrant | Chroma | Astra DB |
|---|---|---|---|---|---|---|
| **License** | Proprietary SaaS | Open-source (BSD) | Open-source (Apache 2.0) | Open-source (Apache 2.0) | Open-source (Apache 2.0) | Proprietary SaaS (Cassandra: Apache 2.0) |
| **Self-hosted** | No | Yes | Yes | Yes | Yes | No |
| **Managed cloud** | Yes (only) | Yes (WCS) | Yes (Zilliz Cloud) | Yes (Qdrant Cloud) | Yes (Chroma Cloud) | Yes (only) |
| **Primary index** | HNSW + IVF+PQ | HNSW, Flat, Dynamic, HFresh | HNSW, IVF variants, DiskANN, SCANN, GPU | HNSW (only dense) | HNSW | SAI (HNSW via JVector) |
| **GPU indexing** | No | No | Yes (CAGRA/NVIDIA) | Yes (v1.16+) | No | No |
| **Disk-based ANN** | Object storage (serverless) | Flat index (disk-backed) | DiskANN | Inline storage (v1.16+) | Object storage (cloud) | SAI on SSTables |
| **Distance metrics** | Cosine, Euclidean, Dot product | Cosine, L2, Dot, Hamming, Manhattan | L2, IP, Cosine, Hamming, Jaccard | Cosine, Dot, L2, Manhattan, Hamming | Cosine, L2, Dot product | Cosine, Dot product, L2 |
| **Hybrid search** | Dense + sparse (keyword+semantic) | BM25 + vector (RRF/relativeScoreFusion) | Dense + sparse (BM25, SPLADE, BGE-M3) | Dense + sparse (BM25/SPLADE) | BM25 + SPLADE (first-class) | ANN + SAI filter predicates |
| **Metadata filtering** | Yes | Yes (rich, per-property indexed) | Yes | Yes (JSON payload, ACORN) | Yes (where clause) | Yes (CQL predicates via SAI) |
| **Storage type** | Cloud/object storage only | Disk + memory (WAL + LSM) | Cloud-native (object storage) + mmap/disk | Disk (RocksDB) + memory | Disk (SQLite+binary) or cloud | Cassandra SSTable (LSM, cloud) |
| **In-memory option** | No (serverless abstraction) | Yes (HNSW loaded to RAM) | Yes (query node memory) | Yes (default) | Yes (EphemeralClient) | No (disk-based) |
| **Lazy / partial loading** | N/A (serverless) | Tenant cold storage tiering | Yes (Milvus 2.6+) | Partial (mmap, on-disk payloads) | Query-aware tiering (cloud) | N/A (managed) |
| **Persistence model** | Managed object storage | WAL + LSM-Tree + HNSW snapshots | Object storage + WAL + log broker | WAL + RocksDB segments | SQLite + binary (local) / object storage (cloud) | Cassandra commit log + SSTables |
| **Sharding** | Auto (serverless) | Per-collection shards across nodes | Distributed query/data/index nodes | N shards per collection (recommended: 12) | Dynamic (cloud only) | Cassandra vnodes (consistent hashing) |
| **Replication** | Managed | Configurable | Configurable | Configurable (factor >= 2 recommended) | N/A (cloud-managed) | Cassandra replication per keyspace |
| **Multi-tenancy** | Namespace-based | Dedicated shard per tenant; hot/cold tiering | Partition key or collection per tenant | Tiered multitenancy (v1.16+); sub-indexes | Tenant/Database/Collection hierarchy | Keyspace/table scoping + RBAC |
| **Auth** | API key, RBAC, SSO | API key, OIDC, RBAC, social login | API key, RBAC, TLS | API key, JWT/RBAC, SSO (cloud) | None (v1.0.x); cloud-managed | Token, SAML SSO, PrivateLink |
| **RBAC** | Yes | Yes | Yes | Yes | No (open-source) | Yes (scoped credentials) |
| **Compliance** | SOC 2 Type II, GDPR | HIPAA (WCS, 2025) | — | — | — | PCI, SOC 2, HIPAA, GDPR |
| **Embedding integrations** | Pinecone hosted models, OpenAI, Cohere | OpenAI, Cohere, HuggingFace, Google, custom; WCS embedding service | OpenAI, Cohere, HuggingFace, BGE, SPLADE, BM25, LangChain | Fastembed (own), any external; 35+ integrations | Sentence Transformers (default), OpenAI, Cohere, Google, OpenCLIP | Astra Vectorize: OpenAI, Azure OpenAI, NVIDIA NeMo, Cohere, HuggingFace, Google |
| **SDK languages** | Python, Node.js, Java, Go | Python, TypeScript, Go, Java | Python, Java, Go, Node.js | Python, TypeScript, Rust, Go, .NET, Java | Python, JavaScript/TypeScript | Python, TypeScript, Java (+ CQL drivers) |
| **Open-source** | No | Yes | Yes | Yes | Yes | No (Cassandra is open) |
| **Unique differentiator** | Serverless-first; hybrid dense+sparse; zero infra | GraphQL + generative search modules + Query Agent | Widest index variety; GPU; lazy loading; time travel; trillion-scale | Rust performance; richest JSON payload filtering; ACORN; tiered multitenancy; Fastembed | Simplest DX; local embeddings; portable; 2025 Rust rewrite | Cassandra backbone; ZCS; real-time updates; Astra Vectorize with NVIDIA NeMo; multi-model |

---

## Notes & Caveats

1. **Pinecone** is the only product with no self-hosted option. All data lives in Pinecone's cloud. The BYOC option on GCP keeps data in the customer's GCP account but is still managed by Pinecone.

2. **Weaviate** is the only database with a **generative search module built directly into the query pipeline** — LLM calls happen server-side within Weaviate, not in a separate application layer.

3. **Milvus** has by far the most index algorithm options (10+), the only production GPU indexing (CAGRA), and the most sophisticated lazy/partial loading semantics (Milvus 2.6+).

4. **Qdrant** is the only database to implement HNSW inline storage (v1.16) — quantized vectors co-located with graph nodes for single disk-read search traversal, giving near-in-memory accuracy with minimal RAM.

5. **Chroma** is the only database with no native authentication in its open-source release (v1.0.x as of 2025). It is best suited for prototyping and small production workloads.

6. **DataStax Astra DB** (post IBM acquisition, May 2025) is the only database that uses Cassandra's wide-column store as a co-equal data model alongside vectors, with true multi-model (tabular + vector + streaming) in a single row.

7. All products support hybrid search (vector + keyword/BM25) as of 2025, though implementation depth varies significantly.

8. Compliance posture: Astra DB (PCI + HIPAA + SOC2 + GDPR) and Pinecone (SOC2 + GDPR) lead; Weaviate Cloud added HIPAA in 2025; Milvus and Qdrant rely on self-managed security controls for compliance.
