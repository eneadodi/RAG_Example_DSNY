from typing import Any, Generator
from tqdm.auto import tqdm
import logging
from datetime import datetime
from models import Paper
from preprocessor import ArXivPreprocessor
from data_loading import get_or_create_database
from sqlalchemy.orm import Session 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Vectorizer:
    """Handles document vectorization and vector storage"""
    
    def __init__(
        self,
        encoder, 
        index,  
        batch_size: int = 5
    ):
        """
        Initialize vectorizer with encoder and storage
        
        Args:
            encoder: Model for generating embeddings
            index: Pinecone index for vector storage
            batch_size: Size of batches for processing
        """
        self.encoder = encoder
        self.index = index
        self.batch_size = batch_size

    def build_chunk(self, title: str, content: str) -> str:
        """Build chunk content with title prefix"""
        return f"# {title}\n{content}"

    def batch_generator(
        self,
        metadata_list: list[dict[str, Any]]
    ) -> Generator[list[dict[str, Any]], None, None]:
        """Generate batches of metadata for processing"""
        for i in range(0, len(metadata_list), self.batch_size):
            yield metadata_list[i:i + self.batch_size]

    def vectorize_and_store(
        self,
        metadata_list: list[dict[str, Any]],
        show_progress: bool = True
    ) -> dict[str, Any]:
        """
        Vectorize documents and store in vector database
        """
        try:
            start_time = datetime.now()
            total_chunks = len(metadata_list)
            processed_chunks = 0
            failed_chunks = 0

            # Process in batches
            batches = list(self.batch_generator(metadata_list))
            
            for batch in tqdm(batches, disable=not show_progress):
                try:
                    # Prepare batch data
                    ids = [item["id"] for item in batch]
                    texts = [
                        self.build_chunk(
                            title=item["title"],
                            content=item["content"]
                        )
                        for item in batch
                    ]
                    
                    # Generate embeddings
                    embeddings = self.encoder(texts)
                    
                    # Prepare metadata - convert to dict format expected by Pinecone
                    metadata_batch = [{
                        'title': item['title'],
                        'content': item['content'],
                        'prechunk_id': item['prechunk_id'],
                        'postchunk_id': item['postchunk_id'],
                        'arxiv_id': item['arxiv_id'],
                        'references': item['references'],
                        'chunk_index': item['chunk_index'],
                        'token_count': item['token_count'],
                    } for item in batch]
                    
                    # Store in vector database
                    vectors_to_upsert = list(zip(
                        ids,
                        embeddings,
                        metadata_batch  
                    ))
                    
                    self.index.upsert(vectors=vectors_to_upsert)
                    
                    processed_chunks += len(batch)
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    failed_chunks += len(batch)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            stats = {
                "total_chunks": total_chunks,
                "processed_chunks": processed_chunks,
                "failed_chunks": failed_chunks,
                "processing_time": duration,
                "chunks_per_second": processed_chunks / duration if duration > 0 else 0
            }

            logger.info(
                f"Vectorization completed:\n"
                f"- Processed chunks: {processed_chunks}\n"
                f"- Failed chunks: {failed_chunks}\n"
                f"- Processing time: {duration:.2f} seconds\n"
                f"- Rate: {stats['chunks_per_second']:.2f} chunks/second"
            )

            return stats

        except Exception as e:
            logger.error(f"Vectorization failed: {str(e)}")
            raise

    def query(
        self,
        text: str,
        top_k: int = 3,
        include_context: bool = True
    ) -> list[dict[str, Any]]:
        """
        Query the vector database
        
        Args:
            text: Query text
            top_k: Number of results to return
            include_context: Whether to include neighboring chunks
        
        Returns:
            list of matching chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.encoder([text])[0]
            
            # Query vector database
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            matches = []
            for match in results["matches"]:
                content = match["metadata"]["content"]
                title = match["metadata"]["title"]
                
                if include_context:
                    # Fetch neighboring chunks
                    pre_id = match["metadata"]["prechunk_id"]
                    post_id = match["metadata"]["postchunk_id"]
                    
                    if pre_id or post_id:
                        context_chunks = self.index.fetch(
                            ids=[id for id in [pre_id, post_id] if id]
                        )["vectors"]
                        
                        # Add context to content
                        if pre_id and pre_id in context_chunks:
                            content = f"{context_chunks[pre_id]['metadata']['content'][-400:]} {content}"
                        if post_id and post_id in context_chunks:
                            content = f"{content} {context_chunks[post_id]['metadata']['content'][:400]}"
                
                matches.append({
                    "title": title,
                    "content": content,
                    "score": match["score"],
                    "metadata": match["metadata"]
                })
            
            return matches

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise


def process_papers_batch(
    papers: list[Paper],
    session: Session,
    preprocessor: ArXivPreprocessor,
    vectorizer: Vectorizer,
    batch_size: int = 5
) -> None:
    """
    Process a batch of papers with preprocessing and optional vectorization
    """
    total_papers = len(papers)
    successful = 0
    failed = 0
    
    logger.info(f"Starting preprocessing of {total_papers} papers...")
    start_time = datetime.now()
    
    # Collect chunks for vectorization
    current_batch_chunks = []
    
    for i, paper in enumerate(papers, 1):
        try:
            chunks_metadata = preprocessor.preprocess_paper(paper)
            
            if chunks_metadata:
                if vectorizer:
                    # chunks_metadata is already in the correct format for vectorization
                    current_batch_chunks.extend(chunks_metadata)
                successful += 1
            
            # Vectorize in batches if needed
            if vectorizer and len(current_batch_chunks) >= batch_size:
                vectorizer.vectorize_and_store(
                    metadata_list=current_batch_chunks,
                    show_progress=True
                )
                current_batch_chunks = []
            
            # Commit database changes in batches
            if i % batch_size == 0:
                try:
                    session.commit()
                    logger.info(f"Processed and committed {i}/{total_papers} papers...")
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error committing batch ending at {i}: {str(e)}")
                    failed += 1
                
        except Exception as e:
            logger.error(f"Failed to process paper {paper.arxiv_id}: {e}")
            failed += 1
            continue
    
    # Process any remaining chunks
    if vectorizer and current_batch_chunks:
        vectorizer.vectorize_and_store(
            metadata_list=current_batch_chunks,
            show_progress=True
        )
    
    # Commit any remaining papers
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Error committing final batch: {str(e)}")
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"""
    Processing completed:
    - Total papers: {total_papers}
    - Successfully processed: {successful}
    - Failed to process: {failed}
    - Time taken: {duration:.2f} seconds
    - Average rate: {total_papers/duration:.2f} papers/second
    """)

def process_existing_papers(
    preprocessor: ArXivPreprocessor,
    vectorizer: Vectorizer,
    db_url: str = 'sqlite:///arxiv_papers.db',
    batch_size: int = 5,
    start_offset: int = 0,
    stop_at: int = 10,
) -> dict:
    """
    Process existing papers from database with preprocessing and optional vectorization
    
    Args:
        preprocessor: ArXivPreprocessor instance
        vectorizer: Optional Vectorizer instance
        db_url: Database URL
        batch_size: Size of batches for processing
        start_offset: Starting offset for processing (useful for resuming)
        
    Returns:
        dict containing processing statistics
    """
        
    Session = get_or_create_database(db_url)
    start_time = datetime.now()
    
    stats = {
        "total_papers": 0,
        "processed_papers": 0,
        "failed_papers": 0,
        "processing_time": 0,
        "papers_per_second": 0
    }
    
    try:
        with Session() as session:
            # Build base query - filter out already processed papers if needed
            query = session.query(Paper)
            
            # Get total count
            total_papers = min(query.count(), stop_at)
            
            if total_papers == 0:
                logger.info("No papers found to process")
                return stats
                
            stats["total_papers"] = total_papers
            logger.info(f"Found {total_papers} papers to process")
            
            processed = 0
            failed = 0
            offset = start_offset
            
            # Process in batches with progress bar
            with tqdm(total=total_papers, initial=start_offset) as pbar:
                while offset < total_papers:
                    try:
                        # Get batch of papers
                        papers_batch = (query
                            .order_by(Paper.id)
                            .offset(offset)
                            .limit(batch_size)
                            .all())
                        
                        if not papers_batch:
                            break
                        
                        # Process batch
                        try:
                            process_papers_batch(
                                papers=papers_batch,
                                session=session,
                                preprocessor=preprocessor,
                                vectorizer=vectorizer,
                                batch_size=batch_size
                            )
                            processed += len(papers_batch)
                            
                        except Exception as batch_error:
                            logger.error(f"Error processing batch at offset {offset}: {batch_error}")
                            failed += len(papers_batch)
                            
                            # Optionally save error information
                            failed_ids = [p.arxiv_id for p in papers_batch]
                            logger.error(f"Failed paper IDs: {failed_ids}")
                        
                        # Update progress
                        pbar.update(len(papers_batch))
                        pbar.set_description(
                            f"Processed: {processed}, Failed: {failed}"
                        )
                        
                    except Exception as query_error:
                        logger.error(f"Error querying batch at offset {offset}: {query_error}")
                        failed += batch_size
                    
                    finally:
                        # Always increment offset to avoid infinite loops
                        offset += batch_size
            
            # Calculate final statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            stats.update({
                "processed_papers": processed,
                "failed_papers": failed,
                "processing_time": duration,
                "papers_per_second": processed / duration if duration > 0 else 0
            })
            
            # Log final summary
            logger.info(f"""
            Processing completed:
            - Total papers: {total_papers}
            - Successfully processed: {processed}
            - Failed to process: {failed}
            - Time taken: {duration:.2f} seconds
            - Average rate: {stats['papers_per_second']:.2f} papers/second
            """)
            
            return stats
            
    except Exception as e:
        logger.error(f"Fatal error in process_existing_papers: {str(e)}")
        raise