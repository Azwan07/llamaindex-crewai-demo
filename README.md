# LlamaIndex + CrewAI Demo

This repository contains a simple starter project that demonstrates:
- LlamaIndex for Retrieval-Augmented Generation (RAG)
- CrewAI for multi-agent workflows
- A small end-to-end Python setup for experimentation

## What is included

- `src/llamaindex_rag.py`: a basic LlamaIndex document ingestion and Q&A example
- `src/crewai_demo.py`: a basic CrewAI multi-agent workflow
- `data/sample.txt`: example content for the RAG pipeline
- `main.py`: runs both examples in one script

## Quick start

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template and add your OpenAI API key:

```bash
cp .env.example .env
```

Then update `.env` with your key:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

4. Run the demo:

```bash
python main.py
```

## Project goals

This repo is designed to help you experiment with:
- document indexing and retrieval with LlamaIndex
- multi-agent task orchestration with CrewAI
- combining retrieval + agentic workflows in a single starter project

## Notes

- This is a minimal demonstration project for learning and prototyping.
- You can expand the data folder with custom documents and adapt the prompts to your use case.
- CrewAI and LlamaIndex can also be combined in more advanced systems where agents retrieve data before acting.
