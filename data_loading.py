import logging
from typing import Any
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from datetime import datetime

from models import Paper,Category,Author,Base



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_or_create_database(db_url: str = 'sqlite:///arxiv_papers.db') -> sessionmaker:
    """
    Create the SQLite database, tables and return a session factory.
    
    Args:
        db_url: Database URL (defaults to SQLite)
    
    Returns:
        sessionmaker: SQLAlchemy session factory
    """
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

def parse_authors(authors_str: str) -> list[dict[str, str]]:
    """
    Parse author string into list of author dictionaries.
    Now handles additional author information if available.
    
    Args:
        authors_str: Comma-separated string of author names
    
    Returns:
        list of dictionaries containing author information
    """
    authors = []
    for name in authors_str.split(','):
        name = name.strip()
        # Basic parsing - could be enhanced to extract email/institution if available
        author_info = {
            'name': name,
            'email': None,
            'institution': None
        }
        authors.append(author_info)
    return authors

def parse_categories(categories_str: str) -> list[dict[str, str]]:
    """
    Parse category string into list of category dictionaries.
    
    Args:
        categories_str: Comma-separated string of category codes
    
    Returns:
        list of dictionaries containing category information
    """
    categories = []
    for code in categories_str.split(','):
        code = code.strip()
        # You might want to maintain a mapping of category codes to full names/descriptions
        category_info = {
            'code': code,
            'name': code,  # Could be mapped to full name
            'description': None
        }
        categories.append(category_info)
    return categories

def get_or_create_author(
    session: Session, 
    author_info: dict[str, str]
) -> Author:
    """
    Get existing author or create new one.
    
    Args:
        session: SQLAlchemy session
        author_info: dictionary containing author information
    
    Returns:
        Author: Author instance
    """
    author = session.query(Author).filter_by(name=author_info['name']).first()
    if not author:
        author = Author(
            name=author_info['name'],
            email=author_info['email'],
            institution=author_info['institution']
        )
        session.add(author)
    return author

def get_paper(
    session: Session, 
    paper_info: dict[str, str],
    get_by="id",
) -> Paper:
    """
    Get existing paper
    
    Args:
        session: SQLAlchemy session
        paper_info: dictionary containing paper information
        get_by: either arxiv_id or title.
    
    Returns:
        paper: Paper instance
    """
    if get_by == "id":
        paper = session.query(Paper).filter_by(arxiv_id=paper_info[get_by]).first()
    elif get_by == "title":
        paper = session.query(Paper).filter_by(title=paper_info[get_by]).first()
    return paper

def get_papers(
    session: Session,
    limit:int = 10,
    all:bool=False
):
    """
    Get existing papers 
    
    Args:
        session: SQLAlchemy session
        limit: how many records to return
        all: if responding ALL papers.
    
    Returns:
        papers: list[Paper] instance
    """
    query = session.query(Paper)
    if all:
        return query.all()
    else:
        return query.limit(limit).all()

def get_or_create_category(
    session: Session, 
    category_info: dict[str, str]
) -> Category:
    """
    Get existing category or create new one.
    
    Args:
        session: SQLAlchemy session
        category_info: dictionary containing category information
    
    Returns:
        Category: Category instance
    """
    category = session.query(Category).filter_by(code=category_info['code']).first()
    if not category:
        category = Category(
            code=category_info['code'],
            name=category_info['name'],
            description=category_info['description']
        )
        session.add(category)
    return category

def load_paper_data(
    session: Session, 
    paper_data: dict[str, Any],
    batch_size: int = 100
) -> Paper | None:
    """
    Load a single paper's data into the database.
    
    Args:
        session: SQLAlchemy session
        paper_data: dictionary containing paper data
        batch_size: Number of records to process before committing
    
    Returns:
        Paper or None: Created paper instance or None if error occurred
    """
    try:
        # Parse and create/get authors
        author_objects = [
            get_or_create_author(session, author_info)
            for author_info in parse_authors(paper_data['authors'])
        ]

        # Parse and create/get categories
        category_objects = [
            get_or_create_category(session, category_info)
            for category_info in parse_categories(paper_data['categories'])
        ]

        # Create paper
        paper = Paper(
            arxiv_id=paper_data['id'],
            title=paper_data['title'],
            summary=paper_data['summary'],
            content=paper_data['content'],
            source=paper_data['source'],
            comment=paper_data.get('comment'),
            journal_ref=paper_data.get('journal_ref'),
            primary_category=paper_data['primary_category'],
            published=paper_data['published'],
            updated=paper_data['updated'],
            authors=author_objects,
            categories=category_objects
        )
        session.add(paper)

        # Store references for later processing
        if paper_data.get('references'):
            paper._temp_citation_ids = list(paper_data['references'].values())

        return paper

    except IntegrityError as e:
        session.rollback()
        logger.error(f"Integrity error loading paper {paper_data['id']}: {str(e)}")
        return None
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error loading paper {paper_data['id']}: {str(e)}")
        return None
    
def load_dataset_todb(
    dataset: list[dict[str, Any]], 
    db_url: str = 'sqlite:///arxiv_papers.db',
    batch_size: int = 100
) -> None:
    """
    Load the entire dataset into the database with batch processing.
    
    Args:
        dataset: list of paper data dictionaries
        db_url: Database URL
        batch_size: Number of records to process before committing
    """
    Session = get_or_create_database(db_url)
    
    with Session() as session:
        total_papers = len(dataset)
        successful_loads = 0
        failed_loads = 0
        
        logger.info(f"Starting to load {total_papers} papers...")
        start_time = datetime.now()

        for i, paper_data in enumerate(dataset, 1):
            paper = load_paper_data(session, paper_data)
            
            if paper:
                successful_loads += 1
            else:
                failed_loads += 1

            # Commit in batches
            if i % batch_size == 0:
                try:
                    session.commit()
                    logger.info(f"Processed {i}/{total_papers} papers...")
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error committing batch ending at {i}: {str(e)}")

        # Commit any remaining records
        try:
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error committing final batch: {str(e)}")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"""
        Loading completed:
        - Total papers processed: {total_papers}
        - Successfully loaded: {successful_loads}
        - Failed to load: {failed_loads}
        - Time taken: {duration:.2f} seconds
        - Average rate: {total_papers/duration:.2f} papers/second
        """)