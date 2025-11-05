# 🏗️ Architecture Diagram

## System Architecture Overview

```mermaid
graph TB
    subgraph "Frontend (React)"
        UI[DogsUI Component]
        WS_CLIENT[WebSocket Client]
        UI --> WS_CLIENT
    end

    subgraph "Backend (FastAPI)"
        API[FastAPI Server]
        WS_ENDPOINT[WebSocket Endpoint<br/>/ws/search]
        SEARCH_ENGINE[CompleteSearchEngine]
        
        API --> WS_ENDPOINT
        WS_ENDPOINT --> SEARCH_ENGINE
    end

    subgraph "Search Pipeline"
        LLM_PARSER[LLM Query Parser<br/>GPT-4o-mini]
        QUERY_ENHANCER[Query Enhancer]
        VECTOR_SEARCH[Vector Search]
        CROSS_ENCODER[Cross-Encoder Reranker<br/>MS-MARCO]
        POST_FILTER[Post-Filter]
        CATEGORIZER[Match Categorizer]
        
        SEARCH_ENGINE --> LLM_PARSER
        LLM_PARSER --> QUERY_ENHANCER
        QUERY_ENHANCER --> VECTOR_SEARCH
        VECTOR_SEARCH --> CROSS_ENCODER
        CROSS_ENCODER --> POST_FILTER
        POST_FILTER --> CATEGORIZER
    end

    subgraph "External Services"
        OPENAI[OpenAI API<br/>Embeddings & LLM]
        PINECONE[Pinecone<br/>Vector Database]
    end

    subgraph "Data Storage"
        INDEX[Pinecone Index<br/>Dog Breed Embeddings]
        JSON_DATA[Enriched Breed Data<br/>JSON Files]
    end

    WS_CLIENT <-->|WebSocket| WS_ENDPOINT
    LLM_PARSER --> OPENAI
    QUERY_ENHANCER --> OPENAI
    VECTOR_SEARCH --> PINECONE
    PINECONE --> INDEX
    CATEGORIZER --> WS_ENDPOINT
    WS_ENDPOINT --> WS_CLIENT

    style UI fill:#5B9BD5
    style SEARCH_ENGINE fill:#991B1B
    style OPENAI fill:#10A37F
    style PINECONE fill:#663399
    style INDEX fill:#663399
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

## System Components

### Frontend Components

```
DogsUI (Main Component)
├── WebSocket Connection Manager
├── Search Input Handler
├── Results Display
│   ├── Card Grid (12 per page)
│   ├── Match Category Badges
│   └── Pagination Controls
└── Carousel (Initial Load)
```

### Backend Components

```
CompleteSearchEngine
├── LLMQueryParser
│   └── Extracts: size, weight, temperament, activity_level, etc.
├── Query Enhancer
│   └── Adds semantic context for embeddings
├── Vector Search
│   └── Pinecone query with metadata filters
├── Cross-Encoder Reranker
│   └── Scores all candidates (100)
├── Post-Filter
│   └── Applies explicit filters
└── Match Categorizer
    └── Detects score drops → Excellent/Good/Fair
```


## Search Pipeline Details

### Stage 1: Query Understanding
- **Input**: Natural language query (e.g., "small apartment dog")
- **Process**: LLM extracts structured filters
- **Output**: `{size: "small", apartment_suitable: true, ...}`

### Stage 2: Query Enhancement
- **Input**: Original query + extracted filters
- **Process**: Adds semantic context
- **Output**: Enhanced query string for embedding

### Stage 3: Embedding Generation
- **Input**: Enhanced query
- **Process**: OpenAI embedding API
- **Output**: 1536-dimensional vector

### Stage 4: Vector Search
- **Input**: Query embedding + metadata filters
- **Process**: Pinecone similarity search
- **Output**: Top 100 candidates with scores

### Stage 5: Cross-Encoder Reranking
- **Input**: 100 candidates + original query
- **Process**: Cross-encoder scores each candidate
- **Output**: Reranked results by relevance score

### Stage 6: Post-Filtering
- **Input**: Reranked results + LLM filters
- **Process**: Applies explicit filters (e.g., temperament_avoid)
- **Output**: Filtered results

### Stage 7: Categorization
- **Input**: Filtered results with scores
- **Process**: Detects drastic score drops to set boundaries
- **Output**: Results tagged as Excellent/Good/Fair

### Stage 8: Result Formatting
- **Input**: Categorized results
- **Process**: Formats for frontend display
- **Output**: 24 results with match categories



## Performance Considerations

- **Vector Search**: Retrieves top 100 candidates for better quality
- **Cross-Encoder**: Reranks all 100 candidates (not just subset)
- **Async Processing**: WebSocket with thread pool for non-blocking operations
- **Pagination**: 24 results split into 2 pages of 12 for better UX
- **Caching**: Embeddings and reranking results cached in memory

## Security

- **API Keys**: Stored in `.env` file (not committed to git)
- **CORS**: Configured for frontend domain
- **Input Validation**: Query sanitization and validation
- **Error Handling**: Comprehensive error handling with logging

