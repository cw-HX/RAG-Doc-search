**Project Overview**

- **Purpose:** A small, dynamic Retrieval-Augmented Generation (RAG) application that lets a user supply article URLs, scrapes and splits the content into chunks, embeds the chunks into a local FAISS vector store using HuggingFace embeddings, and then answers user questions by retrieving relevant chunks and using an LLM to synthesize answers.
- **What it achieves:** Rapid, on-demand QA over arbitrary web articles (added at runtime) without requiring a central document database. Useful for prototyping article question-answering workflows.

**Technologies & Libraries**

- **Language:** Python
- **Web UI:** Streamlit (`streamlit`) — interactive UI and session state
- **LLM interface:** `langchain_openai.ChatOpenAI` wrapper configured in `src/config/config.py` (here pointed at Groq's OpenAI-compatible endpoint)
- **Orchestration / State machine:** `langgraph` (StateGraph) to model the small RAG pipeline as nodes and edges
- **Document loading/splitting:** `langchain_community.document_loaders.WebBaseLoader`, `langchain_text_splitters.RecursiveCharacterTextSplitter`
- **Embeddings:** `langchain_huggingface.HuggingFaceEmbeddings` with `all-MiniLM-L6-v2` (CPU-friendly)
- **Vector store:** FAISS (`faiss-cpu`) via `langchain_community.vectorstores.FAISS`
- **Schemas & validation:** `pydantic` for the `RAGState` model
- **Other tooling:** `python-dotenv` (env vars), common deps in `requirements.txt` (see list generated from pyproject)

**High-level Architecture (files & responsibility)**

- `streamlit_app.py` — Streamlit UI: collects URLs, initializes the system (`initialize_rag`), accepts user questions, displays answers and source excerpts. Uses Streamlit session state and `@st.cache_resource` to cache the initialized RAG stack.
- `src/config/config.py` — Configuration and LLM factory. Reads `XAI_API_KEY` from env and constructs a `ChatOpenAI` client pointing to Groq's endpoint and configured model name.
- `src/document_ingestion/document_processor.py` — Document loader & splitter. Loads web pages (or local files) using LangChain loaders, then splits them into chunks using a `RecursiveCharacterTextSplitter` with configurable `CHUNK_SIZE` and `CHUNK_OVERLAP`.
- `src/vectorstore/vectorstore.py` — Embedding and vector store layer. Uses Hugging Face local embeddings + FAISS. Provides `create_vectorstore(documents)` to build FAISS and `get_retriever()`.
- `src/graph_builder/graph_builder.py` — Builds a `StateGraph` (from `langgraph`) using `RAGState` + nodes from `src/node/reactnode.py` (or `nodes.py`). Adds `retriever` and `responder` nodes and compiles the graph.
- `src/node/reactnode.py` and `src/node/nodes.py` — Node implementations (slightly duplicated):
  - `retrieve_docs` node: invokes the retriever with the question and stores returned docs in `RAGState`.
  - `generate_answer` node: concatenates retrieved docs into a `context`, constructs a prompt, calls `llm.invoke(prompt)` and stores the textual answer in `RAGState`.
- `src/state/rag_state.py` — `RAGState` Pydantic model: fields for `question`, `retrieved_docs`, and `answer`.
- `main.py` — trivial placeholder entrypoint.
- `requirements.txt` — pinned dependency list used to reproduce the environment.

**Detailed Workflow — step-by-step (runtime)**

1. User opens the Streamlit UI (`streamlit run streamlit_app.py`).
2. In the sidebar the user adds one or more article URLs and clicks `Initialize Search Engine`.
3. Streamlit calls `initialize_rag(urls)` (cached via `@st.cache_resource`):
   - `Config.get_llm()` constructs an LLM client using `ChatOpenAI` with model `llama-3.3-70b-versatile` and API key from `XAI_API_KEY`.
   - `DocumentProcessor` is created with `CHUNK_SIZE` and `CHUNK_OVERLAP` from `Config`.
   - `DocumentProcessor.process_urls(active_urls)` runs:
     - `load_documents`: for each URL, `WebBaseLoader` is used to fetch the page and produce a `langchain_core.documents.Document` object(s).
     - `split_documents`: the `RecursiveCharacterTextSplitter` splits large text into overlapping chunks of ~500 tokens (configurable).
   - The resulting list of chunk `Document` objects is passed to `VectorStore.create_vectorstore(documents)`:
     - `HuggingFaceEmbeddings` computes embeddings (model `all-MiniLM-L6-v2`) for each chunk.
     - FAISS index is built via `FAISS.from_documents(...)` and a retriever is created `vectorstore.as_retriever()`.
   - `GraphBuilder(retriever, llm)` is created and `build()` registers two nodes (`retriever` and `responder`) and compiles them into a runnable `StateGraph`.
   - `initialize_rag` returns the compiled graph (`graph_builder`) and the number of indexed chunks.
4. UI becomes interactive. When the user enters a question and submits:
   - Streamlit calls `graph.invoke(RAGState(question=...))` (via `GraphBuilder.run`), which executes the state graph:
     - `retriever` node: calls `retriever.invoke(state.question)` — this returns a list of `Document` objects (the most relevant chunks).
     - `responder` node: concatenates retrieved chunks into a `context` string, constructs a prompt asking the LLM to answer using ONLY the context, then calls `llm.invoke(prompt)`.
     - The LLM response is placed in `RAGState.answer` and the final state is returned.
5. The UI displays the answer and shows short source excerpts from the retrieved documents.

**How retrieval & generation are wired**

- Retrieval: `VectorStore` exposes a retriever object (LangChain-style) wrapping FAISS. The nodes call `retriever.invoke(question)` to get ranked chunk documents.
- Generation: The project uses the configured `ChatOpenAI` wrapper (`Config.get_llm()`) which exposes `invoke(prompt)` returning an object with `.content`. The prompt used is simple: embed the concatenated retrieved text as `Context` and the user question below it, instructing the model to answer from context.

**Configuration & Environment**

- Set environment variables (recommended in a `.env` file):
  - `XAI_API_KEY` — required for `Config.get_llm()` to call the LLM endpoint (Groq/OpenAI-compatible).
- Install dependencies (from repository root):

```bash
python -m pip install -r requirements.txt
```

- Run the app:

```bash
streamlit run streamlit_app.py
```

**Runtime behaviour & performance considerations**

- Embeddings are computed locally using `all-MiniLM-L6-v2` which is CPU-friendly but not as accurate as larger models. FAISS is used in-memory; by default the index is not persisted to disk, so restarting the app will require re-indexing.
- Web scraping with `WebBaseLoader` can fail on sites with heavy JavaScript or anti-bot measures. Consider using more robust loaders or a headless browser if needed.
- LLM calls are proxied to the configured `base_url` (Groq), so latency and cost depend on that provider.
- `@st.cache_resource` caches the initialized RAG stack for the Streamlit session; however, multi-user or multi-session deployments might need a different caching/persistence strategy.

**Security & best practices**

- Never commit API keys. Use `.env` and local environment variables.
- Sanitize or limit user-provided URLs if exposing this app to non-trusted users (to avoid SSRF or scraping abuse).
- For production, consider persisting FAISS indices to disk or a remote vector store and add authentication to the UI.

**Limitations and opportunities for improvement**

- Persist vectors to disk or external vector DB (Milvus, Pinecone, etc.) to avoid reindexing.
- Add rate-limiting, URL validation, and retry/backoff for HTTP fetches.
- Improve prompt engineering, use chain-of-thought or structured prompts, or call the LLM with streaming for better UX.
- Add provenance: include exact source URLs and character offsets in returned `Document` metadata to make references auditable.
- Replace simple concatenation context with an answer synthesis chain (e.g., use a summarization step for many retrieved docs).

**Developer notes & file pointers**

- UI logic: `streamlit_app.py`
- LLM config: `src/config/config.py`
- Document ingestion: `src/document_ingestion/document_processor.py`
- Vector & embeddings: `src/vectorstore/vectorstore.py`
- Graph orchestration: `src/graph_builder/graph_builder.py`
- Nodes: `src/node/reactnode.py` and `src/node/nodes.py`
- State: `src/state/rag_state.py`


---

If you want, I can now:
- Commit `TECHNICAL.md` and push it to `origin/main` (I can do that now), or
- Expand sections (e.g., add sequence diagrams, sample prompts, or a persisted FAISS example).

