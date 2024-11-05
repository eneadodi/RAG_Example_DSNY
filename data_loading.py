from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

def create_database():
    """Create the SQLite database and tables"""
    engine = create_engine('sqlite:///arxiv_papers.db')
    Base.metadata.create_all(engine)
    return engine

def parse_authors(authors_str: str) -> List[str]:
    """Split authors string into individual names"""
    return [name.strip() for name in authors_str.split(',')]

def parse_categories(categories_str: str) -> List[str]:
    """Split categories string into individual codes"""
    return [cat.strip() for cat in categories_str.split(',')]

def load_paper_data(session: Session, paper_data: Dict[str, Any]) -> None:
    """Load a single paper's data into the database"""
    try:
        # Create or get authors
        author_objects = []
        for author_name in parse_authors(paper_data['authors']):
            author = session.query(Author).filter_by(name=author_name).first()
            if not author:
                author = Author(name=author_name)
                session.add(author)
            author_objects.append(author)

        # Create or get categories
        category_objects = []
        for category_code in parse_categories(paper_data['categories']):
            category = session.query(Category).filter_by(code=category_code).first()
            if not category:
                category = Category(code=category_code, name=category_code)  # You might want to map codes to full names
                session.add(category)
            category_objects.append(category)

        # Create paper
        paper = Paper(
            arxiv_id=paper_data['id'],
            title=paper_data['title'],
            summary=paper_data['summary'],
            content=paper_data['content'],
            source=paper_data['source'],
            comment=paper_data['comment'],
            journal_ref=paper_data['journal_ref'],
            primary_category=paper_data['primary_category'],
            published=paper_data['published'],
            updated=paper_data['updated'],
            authors=author_objects,
            categories=category_objects
        )
        session.add(paper)

        # Handle references
        if paper_data.get('references'):
            for ref_id in paper_data['references'].values():
                cited_paper = session.query(Paper).filter_by(arxiv_id=ref_id).first()
                if cited_paper:
                    paper.citations.append(cited_paper)

        session.commit()
    except IntegrityError as e:
        session.rollback()
        print(f"Error loading paper {paper_data['id']}: {str(e)}")
    except Exception as e:
        session.rollback()
        print(f"Unexpected error loading paper {paper_data['id']}: {str(e)}")

def load_dataset(dataset: List[Dict[str, Any]]) -> None:
    """Load the entire dataset into the database"""
    engine = create_database()
    
    with Session(engine) as session:
        for paper_data in dataset:
            load_paper_data(session, paper_data)

# Usage example:
# load_dataset(dataset)