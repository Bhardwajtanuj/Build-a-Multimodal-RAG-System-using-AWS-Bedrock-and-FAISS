# Multimodal RAG System with AWS Bedrock & FAISS

A Retrieval-Augmented Generation (RAG) system that supports **text and images** using:
- **AWS Bedrock** for embeddings (Nova Multimodal or Titan) and generation (Claude)
- **FAISS** for fast vector similarity search

## Features

- **Multimodal embeddings**: Text and images in a unified vector space (Nova Multimodal)
- **Text fallback**: Uses Amazon Titan Embeddings when Nova is not available
- **Document ingestion**: PDF, TXT, MD, DOCX with automatic chunking
- **Image ingestion**: JPG, PNG with optional captions
- **Text & image queries**: Query by text or by image (e.g., "find similar images")
- **Streamlit UI**: Interactive web interface

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Documents      │     │  AWS Bedrock     │     │   FAISS     │
│  Images         │ ──► │  Embeddings      │ ──► │   Vector    │
│                 │     │  (Nova/Titan)    │     │   Store     │
└─────────────────┘     └──────────────────┘     └──────┬──────┘
                                                        │
┌─────────────────┐     ┌──────────────────┐            │
│  User Query     │ ──► │  Embed Query     │ ───────────┤
│  (text/image)   │     │                  │            │
└─────────────────┘     └──────────────────┘            │
                                                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Generated      │ ◄── │  AWS Bedrock     │ ◄── │  Top-K      │
│  Answer         │     │  Claude LLM      │     │  Retrieved  │
└─────────────────┘     └──────────────────┘     └─────────────┘
```

## Prerequisites

- Python 3.10+
- AWS account with Bedrock access
- AWS credentials configured (`~/.aws/credentials` or env vars)

### Enable Models in Bedrock

1. Go to [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. **Model access** → Enable:
   - **Amazon Nova Multimodal Embeddings** (us-east-1) for full multimodal support
   - **Amazon Titan Embeddings G1 - Text** as fallback
   - **Anthropic Claude 3 Sonnet** for generation

## Installation

```bash
cd "RAG System using AWS"
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and set:

```bash
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

Or rely on default AWS credential chain.

## Usage

### 1. Ingest Documents & Images

Place files in:
- `data/documents/` – PDF, TXT, MD, DOCX
- `data/images/` – JPG, PNG

Then run:

```bash
python main.py ingest --all
```

Or specify paths:

```bash
python main.py ingest --documents doc1.pdf doc2.txt --images img1.jpg
```

### 2. Query via CLI

```bash
# Text query
python main.py query "What is AWS Bedrock?"

# Image-based query (find similar content)
python main.py query "What is in this image?" --image path/to/image.jpg

# Custom top-k
python main.py query "Explain FAISS" --top-k 10
```

### 3. Streamlit App

```bash
python main.py app
# or
streamlit run app.py
```

Open http://localhost:8501 to:
- Ingest documents and images
- Query by text or image
- View retrieved sources and generated answers

## Project Structure

```
RAG System using AWS/
├── app.py              # Streamlit UI
├── main.py             # CLI
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py       # Configuration
│   ├── embeddings.py   # Bedrock embeddings (Nova/Titan)
│   ├── vector_store.py # FAISS wrapper
│   ├── ingest.py       # Document/image ingestion
│   └── rag.py          # RAG pipeline
├── data/
│   ├── documents/      # PDF, TXT, MD, DOCX
│   └── images/         # JPG, PNG
└── faiss_index/        # Persisted FAISS index
```

## Embedding Models

| Model | Type | Region | Use Case |
|-------|------|--------|----------|
| Nova Multimodal | Text + Image | us-east-1 | Full multimodal RAG |
| Titan Embeddings v2 | Text only | Multiple | Fallback when Nova unavailable |

