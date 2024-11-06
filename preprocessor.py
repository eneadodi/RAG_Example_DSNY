

from typing import Any
import re
import logging
import unicodedata
from datetime import datetime
from semantic_chunkers import StatisticalChunker,ConsecutiveChunker
from semantic_router.splitters import RollingWindowSplitter
from sqlalchemy.orm import Session
from models import Paper
from data_loading import get_or_create_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalChunkerWrapper:
    """
    Wrapper class to make StatisticalChunker output consistent format
    """
    def __init__(self, encoder):
        self.chunker = StatisticalChunker(encoder=encoder)
    
    def __call__(self, docs: list[str]) -> list[dict[str, Any]]:
        splits = self.chunker(docs)
        
        chunks = []
        for split in splits:
            # Join the split texts if they're in a list
            text = ' '.join(split.splits) if isinstance(split.splits, list) else split.splits
            
            chunk = {
                'text': text,
                'token_count': split.token_count if split.token_count is not None else 0,
                'score': split.triggered_score
            }
            chunks.append(chunk)
            
        return chunks
    
class RollingWindowWrapper:
    """
    Wrapper class to make RollingWindowSplitter compatible with the StatisticalChunker 
    and ConsecutiveChunker interface
    """
    def __init__(self, encoder, min_split_tokens=100, max_split_tokens=500, window_size=2):
        self.splitter = RollingWindowSplitter(
            encoder=encoder,
            dynamic_threshold=True,
            min_split_tokens=min_split_tokens,
            max_split_tokens=max_split_tokens,
            window_size=window_size,
            plot_splits=False,  # Can be made configurable if needed
            enable_statistics=True
        )
    
    def __call__(self, docs: list[str]) -> list[dict[str, Any]]:
        """
        Convert documents into chunks using RollingWindowSplitter
        
        Args:
            docs: List of document strings to chunk
            
        Returns:
            List of dictionaries containing chunk information in the format:
            [{'text': str, 'token_count': int, 'score': float}, ...]
        """
        all_chunks = []
        
        for doc in docs:
            # Get splits for current document
            splits = self.splitter([doc])
            
            # Convert each DocumentSplit into the expected format
            for split in splits:
                chunk = {
                    'text': split.content,
                    'token_count': split.token_count,
                    'score': split.triggered_score
                }
                all_chunks.append(chunk)
                
        return all_chunks
    

class ArXivPreprocessor:
    def __init__(self, encoder,speed=0):
        if speed == 0:
            self.chunker = StatisticalChunkerWrapper(encoder=encoder)
        elif speed == 1:
            self.chunker = RollingWindowWrapper(
                encoder=encoder,
                min_split_tokens=100,
                max_split_tokens=500,
                window_size=2
            )
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}"\'`-]', ' ', text)
        return ' '.join(text.split()).strip()

    def build_chunk_metadata(
        self,
        paper: Paper,
        chunks: list[dict],
    ) -> list[dict[str, Any]]:
        """
        Build metadata for document chunks
        
        Args:
            paper: Paper model instance
            chunks: list of chunks from StatisticalChunker
        
        Returns:
            list of chunk metadata dictionaries
        """
        metadata = []
        for i, chunk in enumerate(chunks):
            #print
            # Get neighboring chunks
            prechunk_id = "" if i == 0 else f"{paper.arxiv_id}#{i-1}"
            postchunk_id = "" if i+1 == len(chunks) else f"{paper.arxiv_id}#{i+1}"
            
            metadata.append({
                "id": f"{paper.arxiv_id}#{i}",
                "title": paper.title,
                "content": chunk['text'],
                "prechunk_id": prechunk_id,
                "postchunk_id": postchunk_id,
                "arxiv_id": paper.arxiv_id,
                "references": [cited.arxiv_id for cited in paper.citations],
                "chunk_index": i,
                "token_count": chunk.get('token_count'),
                "semantic_score": chunk.get('score', 0.0)
            })
        
        return metadata

    def build_chunk_content(self, title: str, content: str) -> str:
        """Build content with title prefix"""
        return f"# {title}\n{content}"

    def preprocess_paper(self, paper: Paper) -> list[dict] | None :
        """
        Preprocess a paper and update its content and metadata
        
        Args:
            paper: Paper model instance
            
        Returns:
            Optional[list[dict]]: List of chunk metadata if successful
        """
        try:
            # Clean text fields
            if not paper.content:
                logger.warning(f"No content for paper {paper.arxiv_id}, skipping preprocessing")
                return None
            
            title = self.clean_text(paper.title)
            summary = self.clean_text(paper.summary)
            content = self.clean_text(paper.content)
            

            # Process content with semantic chunking
            chunks = self.chunker(docs=[content])
            # Build chunk metadata
            chunks_metadata = self.build_chunk_metadata(paper, chunks)
            
            # Process chunks with titles and combine
            processed_content = []
            for chunk_meta in chunks_metadata:
                chunk_content = self.build_chunk_content(
                    title=title,
                    content=chunk_meta['content']
                )
                processed_content.append(chunk_content)
            
            # Update paper with processed content and metadata
            paper.chunk_count = len(chunks)
            paper.total_tokens = sum(chunk['token_count'] for chunk in chunks_metadata)
            
            logger.info(f"Successfully preprocessed paper {paper.arxiv_id} into {paper.chunk_count} chunks")
            
            return chunks_metadata
            
        except Exception as e:
            logger.error(f"Error preprocessing paper {paper.arxiv_id}: {e}")
            raise

