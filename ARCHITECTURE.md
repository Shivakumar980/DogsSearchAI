# 🏗️ Architecture Diagram

## System Architecture Overview

```mermaid
graph TB
    %% Styling
    classDef ingestNode fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    classDef searchNode fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:#fff
    classDef serviceNode fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff
    classDef dbNode fill:#8b5cf6,stroke:#6d28d9,stroke-width:3px,color:#fff
    classDef userNode fill:#f59e0b,stroke:#d97706,stroke-width:3px,color:#fff
    
    %% --- INGESTION PIPELINE ---
    subgraph INGEST["DATA INGESTION PIPELINE"]
        direction LR
        A1[Clean Data]:::ingestNode
        A2[Map Size Categories]:::ingestNode
        A3[Generate Text Formats]:::ingestNode
        A4[Generate Embeddings]:::ingestNode
        A5[Index to Pinecone]:::ingestNode
        
        A1 --> A2 --> A3 --> A4 --> A5
    end
    
    %% --- MIDDLE LAYER ---
    subgraph SERVICES["OPENAI SERVICES"]
        direction TB
        EMB[Embeddings API<br/>text-embedding-3-small]:::serviceNode
        LLM[LLM API<br/>gpt-4o-mini]:::serviceNode
    end
    
    subgraph DB["VECTOR DATABASE"]
        direction TB
        PC[(Pinecone Index<br/>1536-d vectors<br/>+ metadata)]:::dbNode
    end
    
    %% --- USER ---
    USER[User]:::userNode
    
    %% --- SEARCH PIPELINE ---
    subgraph SRCH["SEARCH PIPELINE"]
        direction LR
        S1[LLM Query Parser]:::searchNode
        S2[Query Enhancer]:::searchNode
        S3[Vector Search]:::searchNode
        S4[Cross-Encoder Rerank]:::searchNode
        S5[Post Filter]:::searchNode
        S6[Match Categorizer]:::searchNode
        
        S1 --> S2 --> S3 --> S4 --> S5 --> S6
    end
    
    %% --- DATA FLOWS ---
    A4 -.->|batch embeddings| EMB
    A5 ==>|upsert vectors| PC
    
    USER ==>|search query| S1
    S1 -.->|parse filters| LLM
    S2 -.->|query embeddings| EMB
    S3 <===>|retrieve| PC
    S6 ==>|results| USER
    
    %% Vertical flow
    INGEST ==> DB
    DB ==> SRCH
    INGEST -.-> SERVICES
    SERVICES -.-> SRCH
```


## Ingestion Pipeline Flow

```mermaid
sequenceDiagram
    participant Admin
    participant Pipeline
    participant DogsAPI
    participant OpenAI
    participant Pinecone
    participant JSON

    Admin->>Pipeline: Run ingestion pipeline
    
    Note over Pipeline: Stage 1: Fetch
    Pipeline->>DogsAPI: GET /breeds
    DogsAPI-->>Pipeline: Raw breed data
    
    Note over Pipeline: Stage 2: Clean
    Pipeline->>Pipeline: Remove duplicates<br/>Handle missing fields<br/>Normalize data
    
    Note over Pipeline: Stage 3: Map Sizes
    Pipeline->>Pipeline: Map weight to categories<br/>(tiny, small, medium, large, giant)
    
    Note over Pipeline: Stage 4: Generate Text
    Pipeline->>Pipeline: Generate conversational text<br/>Generate structured text
    
    Note over Pipeline: Stage 5: Generate Embeddings
    Pipeline->>OpenAI: text-embedding-3-small<br/>(batch of 100)
    OpenAI-->>Pipeline: Embeddings (1536-dim)
    
    Note over Pipeline: Stage 6: Index to Pinecone
    Pipeline->>Pinecone: Upsert vectors + metadata
    Pinecone-->>Pipeline: Indexed successfully
    
    Note over Pipeline: Stage 7: Save JSON
    Pipeline->>JSON: Save enriched data
    JSON-->>Pipeline: Saved with timestamp
    
    Pipeline-->>Admin: ✅ Pipeline complete
```

## Search Pipeline Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant LLM_Parser
    participant OpenAI
    participant Pinecone
    participant CrossEncoder
    participant Categorizer

    User->>Frontend: Enter search query
    Frontend->>Backend: WebSocket: search request
    
    Backend->>LLM_Parser: Parse natural language query
    LLM_Parser->>OpenAI: GPT-4o-mini API call
    OpenAI-->>LLM_Parser: Structured filters
    LLM_Parser-->>Backend: Filters extracted
    
    Backend->>Backend: Enhance query with context
    Backend->>OpenAI: Generate embedding
    OpenAI-->>Backend: Query embedding
    
    Backend->>Pinecone: Vector search (top 100)
    Pinecone-->>Backend: 100 candidates
    
    Backend->>CrossEncoder: Rerank all 100 candidates
    CrossEncoder-->>Backend: Reranked results
    
    Backend->>Backend: Apply post-filters
    Backend->>Categorizer: Categorize by score drops
    Categorizer-->>Backend: Tagged results (24)
    
    Backend->>Frontend: WebSocket: Results with categories
    Frontend->>User: Display results with badges
```

## Component Architecture

```mermaid
graph LR
    subgraph "Frontend Layer"
        A[React App] --> B[DogsUI Component]
        B --> C[WebSocket Handler]
        B --> D[Pagination Logic]
        B --> E[Card Display]
    end

    subgraph "Backend Layer"
        F[FastAPI App] --> G[WebSocket Handler]
        G --> H[CompleteSearchEngine]
        H --> I[LLMQueryParser]
        H --> J[Vector Search]
        H --> K[CrossEncoder]
        H --> L[PostFilter]
        H --> M[ResultFormatter]
    end

    subgraph "Data Layer"
        N[Pinecone Index] --> O[Embeddings]
        P[Enriched JSON] --> Q[Breed Metadata]
    end

    subgraph "External APIs"
        R[OpenAI API]
        S[Pinecone API]
    end

    C --> G
    I --> R
    J --> S
    K --> R
    S --> N
    M --> B

    style A fill:#61DAFB
    style F fill:#009688
    style R fill:#10A37F
    style S fill:#663399
```

## Data Flow Diagram

```mermaid
flowchart TD
    START[User Query] --> PARSE[LLM Query Parser]
    PARSE --> FILTERS[Extract Filters]
    FILTERS --> ENHANCE[Query Enhancement]
    ENHANCE --> EMBED[Generate Embedding]
    EMBED --> VECTOR[Vector Search]
    VECTOR --> TOP100[Top 100 Candidates]
    TOP100 --> RERANK[Cross-Encoder Rerank]
    RERANK --> SCORED[Scored Results]
    SCORED --> FILTER[Post-Filter]
    FILTER --> CATEGORIZE[Match Categorization]
    CATEGORIZE --> RESULT[24 Results with Categories]
    RESULT --> DISPLAY[Frontend Display]

    style START fill:#E8F4FD
    style RESULT fill:#4ADE80
    style DISPLAY fill:#5B9BD5
```



