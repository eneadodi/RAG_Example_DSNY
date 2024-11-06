# Pre-Screening Questions for LLM Data Engineer - DSNY

## Submittal Questions

### 1. Data Ingestion
- Provide a dataset (e.g., JSON, CSV, or unstructured text files) that includes a mix of structured and unstructured data
- Ask the candidate to create a pipeline to load this data into a database of their choice, ensuring the schema is optimized for querying

### 2. Data Preprocessing
- The data may contain noise or require transformation (e.g., text cleaning, parsing nested JSON, handling missing values)
- The candidate should demonstrate how they preprocess the data for efficient storage and later retrieval

### 3. Vectorization
- Using a pre-trained language model or embeddings model, ask the candidate to convert the unstructured text into embeddings
- Store these embeddings in a vector storage solution of their choice, ensuring the pipeline can handle batch processing for larger datasets

### 4. Query and Retrieve
- Create a simple API or script that allows querying based on a given text prompt
- The query should retrieve similar embeddings from the vector store and return the corresponding records from the database
- Include a use case for Retriever-Augmented Generation (RAG), where the retrieved data is used to generate a summary or response based on the query

### 5. Documentation
- The candidate should document their code, the thought process behind their design choices, and any trade-offs they considered (e.g., schema design, vector storage approach, etc.)

### 6. Bonus
- Implement monitoring or logging for the data pipeline to track the data flow and identify potential bottlenecks
- Optimize the pipeline for scalability, such as handling larger files or parallel processing

---

Everything is covered in the .py files as well as a walkthrough on the jupyter notebook `simple-rag.ipynb`

Thank you!  
Enea Dodi