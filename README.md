# 🐕 Dog Breed Finder

An intelligent AI-powered dog breed search application that helps users find their perfect canine companion using natural language queries, semantic search, and cross--encoder reranking.

# Search bar and Results

<img width="1432" height="852" alt="image" src="https://github.com/user-attachments/assets/6b2abf30-97f6-41b4-a8ac-68a7b04cec75" />

<img width="1440" height="855" alt="image" src="https://github.com/user-attachments/assets/e0ad4dfe-1556-4c67-9534-41b48f35abef" />


## ✨ Features

- **Natural Language Search**: Describe what you're looking for in plain English
- **AI-Powered Understanding**: LLM-based query parsing extracts filters and requirements
- **Semantic Search**: Vector embeddings for finding breeds by meaning, not just keywords
- **Intelligent Reranking**: Cross-encoder model scores and ranks results by relevance
- **Match Quality Categories**: Results tagged as Excellent, Good, or Fair Match based on score distribution
- **Real-time Search**: WebSocket-based communication for instant results
- **Pagination**: 24 results per search, displayed in pages of 12
- **Beautiful UI**: Modern, responsive design with gradient theme

## 🏗️ Architecture

For detailed architecture diagrams and system design, see [ARCHITECTURE.md](./ARCHITECTURE.md)

### Overview
- **Backend**: FastAPI server with WebSocket endpoints
- **Frontend**: React application with real-time search
- **AI Services**: OpenAI (embeddings + LLM) and Pinecone (vector database)
- **Search Pipeline**: Multi-stage search with LLM parsing, vector search, reranking, and categorization

### Backend
- **FastAPI**: RESTful API and WebSocket server
- **Pinecone**: Vector database for semantic search
- **OpenAI**: Embeddings generation (text-embedding-3-small)
- **Cross-Encoder**: MS-MARCO model for result reranking
- **LLM Query Parser**: GPT-4o-mini for understanding natural language queries

### Frontend
- **React**: User interface with Vite
- **WebSocket**: Real-time search communication
- **Responsive Design**: Mobile and desktop support

## 📋 Prerequisites

- **Python 3.13+** (or 3.9+)
- **Node.js 18+** and npm
- **API Keys**:
  - OpenAI API key
  - Pinecone API key
  - Pinecone index name

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <https://github.com/Shivakumar980/DogsSearchAI>
cd dog-breed-search
```

### 2. Backend Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
cd backend
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in `backend/config/`:

```bash
cd backend/config
touch .env
```

Add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_pinecone_index_name
```

### 4. Run Data Ingestion (First Time Only)

If you haven't populated your Pinecone index yet:

```bash
cd backend
python main.py
```

This will:
- Fetch dog breed data
- Generate embeddings
- Index data to Pinecone
- Create enriched breed data JSON files

### 5. Start Backend Server

```bash
cd backend
source ../venv/bin/activate  # If not already activated
python app.py
```

The backend server will start on `http://localhost:8000`

### 6. Frontend Setup

Open a new terminal:

```bash
cd frontend
npm install
```

### 7. Start Frontend

```bash
npm run dev
```

The frontend will start on `http://localhost:5173` (or another port if 5173 is in use)

## 🔍 Sample Search Queries

Try these example queries to see the search engine in action:

### Size & Weight Queries
- `"small apartment dog"`
- `"light weight dogs"`
- `"large protective dog"`
- `"medium sized family dog"`

### Temperament Queries
- `"friendly playful dog"`
- `"calm quiet dog for elderly"`
- `"energetic hiking companion"`
- `"gentle dog good with kids"`

### Lifestyle Queries
- `"dog for first-time owner"`
- `"apartment suitable dog"`
- `"dog that won't bark at neighbors"`
- `"low maintenance dog"`

### Activity Level Queries
- `"lazy dog for couch potato"`
- `"high energy running partner"`
- `"moderate activity family dog"`

### Special Requirements
- `"hypoallergenic dog"`
- `"protection dog"`
- `"herding dog"`
- `"hunting companion"`

### Complex Queries
- `"small friendly dog for apartment living with kids"`
- `"energetic hiking companion good with kids"`
- `"dog for elderly person with limited mobility"`
- `"I work long hours, need independent dog"`

## 📁 Project Structure

```
dog-breed-search/
├── backend/
│   ├── app.py                    # FastAPI server with WebSocket endpoints
│   ├── complete_search_engine.py # Main search engine logic
│   ├── llm_query_parser.py      # LLM-based query parsing
│   ├── ingestion_pipeline.py    # Data ingestion and indexing
│   ├── search_cli.py            # Command-line search interface
│   ├── main.py                  # Entry point for ingestion
│   ├── config/
│   │   └── .env                 # Environment variables (create this)
│   ├── data/                    # Generated enriched breed data
│   ├── logs/                    # Server logs
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DogsUI.jsx       # Main search component
│   │   │   └── DogsUI.css       # Component styles
│   │   ├── App.jsx              # Root component
│   │   └── main.jsx             # Entry point
│   ├── public/
│   │   └── images/              # Carousel images
│   ├── package.json             # Node dependencies
│   └── vite.config.js           # Vite configuration
│
├── venv/                        # Python virtual environment
└── README.md                    # This file
```

## 🔧 Technologies Used

### Backend
- **FastAPI** - Modern web framework
- **Pinecone** - Vector database
- **OpenAI** - Embeddings and LLM
- **sentence-transformers** - Cross-encoder reranking
- **uvicorn** - ASGI server
- **websockets** - Real-time communication

### Frontend
- **React** - UI library
- **Vite** - Build tool
- **WebSocket API** - Real-time communication

## 🎯 Search Pipeline

1. **Query Parsing**: LLM extracts structured filters from natural language
2. **Query Enhancement**: Adds semantic context for better embeddings
3. **Vector Search**: Retrieves top 100 candidates from Pinecone
4. **Cross-Encoder Reranking**: Scores all 100 candidates for relevance
5. **Post-Filtering**: Applies any explicit filters
6. **Categorization**: Tags results as Excellent/Good/Fair based on score distribution
7. **Results**: Returns top 24 results with match categories

## 📊 Match Categories

Results are automatically categorized based on score distribution:

- **Excellent Match** ⭐ (Green): Top results with highest relevance scores
- **Good Match** ✓ (Blue): Strong matches worth considering
- **Fair Match** ○ (Amber): Relevant but may not be ideal

Categories are determined by detecting drastic drops in score distribution, ensuring natural groupings.

## 🛠️ Development

### Backend Development

```bash
cd backend
source ../venv/bin/activate
python app.py  # Starts server on http://localhost:8000
```

### Frontend Development

```bash
cd frontend
npm run dev  # Starts dev server (usually http://localhost:5173)
```

### Run Ingestion Pipeline

```bash
cd backend
source ../venv/bin/activate
python main.py
```

### Test Search via CLI

```bash
cd backend
source ../venv/bin/activate
python search_cli.py
```

## 📝 Environment Variables

Required environment variables in `backend/config/.env`:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=dog-breeds
```

## 👥 Contributors

[Shivakumar Machidi]

