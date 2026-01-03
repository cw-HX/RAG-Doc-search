RAG-SearchAPP
=============

Quick overview
--------------

RAG-SearchAPP is a small Retrieval-Augmented Generation (RAG) demo that lets you ingest documents, build a vector-based index/graph, and run semantic search via a Streamlit UI. The project demonstrates document ingestion, vector storage, graph construction, and an interactive frontend for querying.

Repository structure
--------------------

- **streamlit_app.py**: Streamlit application entrypoint (UI + query loop).
- **main.py**: Optional runner or project entry (if present).
- **requirements.txt / pyproject.toml**: Python dependencies.
- **data/**: Example data sources (e.g., `url.txt`).
- **src/**: Application source code organized by responsibility:
	- **config/**: configuration helpers (`config.py`).
	- **document_ingestion/**: document ingestion and preprocessing (`document_processor.py`).
	- **graph_builder/**: graph construction utilities (`graph_builder.py`).
	- **node/**: node definitions and React-like node helpers (`nodes.py`, `reactnode.py`).
	- **state/**: RAG application state management (`rag_state.py`).
	- **vectorstore/**: vector store integration (`vectorstore.py`).

Key concepts
------------

- Document ingestion: turn raw documents or URLs into text chunks and metadata.
- Vector store: embed text chunks and store vectors for nearest-neighbor search.
- Graph builder: create and manage a lightweight graph of nodes to represent relationships between chunks.
- Streamlit UI: interactive interface for querying the vector store and displaying results.

Setup
-----

1. Create a Python virtual environment and activate it (Windows PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Provide required secrets as environment variables. Typical variables used by this project:

- `OPENAI_API_KEY` (or other embedding/model API keys) — set in your shell or a `.env` loader used by `config.py`.

Running the app
---------------

Start the Streamlit UI:

```powershell
streamlit run streamlit_app.py
```

Or run `python main.py` if that script is present and wired as an entrypoint.

Adding documents
----------------

- Place URLs or file references in `data/url.txt` or adapt the ingestion scripts in `src/document_ingestion/document_processor.py`. Run the ingestion flow to populate the vector store.

Development notes
-----------------

- Code is organized for clarity rather than production scalability — good for learning and quick experiments.
- If you change embedding or model provider, update `src/vectorstore/vectorstore.py` and `src/config/config.py` accordingly.
- The `graph_builder` module demonstrates a simple relationship graph; extend it to add richer metadata or persistence.

Contributing
------------

- File issues or open pull requests. Keep changes small and focused.
- Run linters and tests (if added) before submitting changes.

License
-------

This repository does not include an explicit license file. Add one (for example `LICENSE`) if you intend to open-source the project.

Contact / Next steps
--------------------

- Ask for help updating deployment instructions, adding CI, or wiring a hosted vector DB (e.g., Pinecone, Weaviate) if needed.
