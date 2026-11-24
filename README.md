# RAG-for-unstructured-data
This repo aims to create a RAG (Retrieval-Augmented Generation) for unstructured data

#WorkFlow

1. Intelligent Ingestion (Docling) and  Merging of chained tables ----> Output: a rich Markdown document (or JSON) where tables are perfectly formatted and section headers are preserved;
2. Save Images ----> Saving an image of each table to disk (one image per page for multi-page tables), with the file path included in the metadata.
3. Document-Native Chunking ----> Output: through Markdown Splitter (like LangChain's MarkdownHeaderTextSplitter) divide Markdown document into smaller "chunks";
4. Embedding (Vectorization) ----> Output: The embedding model translates the meaning of text into mathematical vectors;
5. Vector Database (Storage) ----> Output: The vectors (and the original Markdown text they represent) are saved in a database.
6. Retrieval and Generation (The User Query):
   1. Search (Retriever): The pipeline converts the question into a vector and searches the Vector Database for the closest mathematical matches. It pulls out the top 5 most relevant Markdown chunks;
   2. Re-ranking: a "Re-ranker" evaluates those 5 chunks to guarantee the absolute best information is prioritized;
   3. Generation (LLM): The pipeline sends those chunks, along with the user's question, to the LLM;
   4. Output: The LLM reads the Markdown table and generates a perfect, conversational answer, citing the exact page number it found the data on;

#Web app

Requires [Ollama](https://ollama.com) with the local model (`ollama pull qwen3.5:4b`). Then:

```
uv run uvicorn src.webapp.main:app --host 127.0.0.1 --port 8000
```

and open http://127.0.0.1:8000. Upload a PDF, wait for it to be processed and ask questions about it: every answer cites the pages it comes from.
The app has no authentication: do not expose it to the internet as it is. While it is running, the scripts cannot open the same Qdrant index.
