# RAG-for-unstructured-data
This repo aims to create a RAG (Retrieval-Augmented Generation) for unstructured data

#WorkFlow

1. Intelligent Ingestion (Docling) ----> Output: a rich Markdown document (or JSON) where tables are perfectly formatted and section headers are preserved;
2. Document-Native Chunking ----> Output: through Markdown Splitter (like LangChain's MarkdownHeaderTextSplitter) divide Markdown document into smaller "chunks";
3. Embedding (Vectorization) --_-> Output: The embedding model translates the meaning of text into mathematical vectors;
4. Vector Database (Storage) ----> Output: The vectors (and the original Markdown text they represent) are saved in a database.
5. Retrieval and Generation (The User Query):
   1. Search (Retriever): The pipeline converts the question into a vector and searches the Vector Database for the closest mathematical matches. It pulls out the top 5 most relevant Markdown chunks;
   2. Re-ranking: a "Re-ranker" evaluates those 5 chunks to guarantee the absolute best information is prioritized;
   3. Generation (LLM): The pipeline sends those chunks, along with the user's question, to the LLM;
   4. Output: The LLM reads the Markdown table and generates a perfect, conversational answer, citing the exact page number it found the data on;
