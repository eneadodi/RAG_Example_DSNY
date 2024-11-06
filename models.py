from sqlalchemy.orm import relationship, DeclarativeBase,Session
from sqlalchemy import Column, DateTime, MetaData, event, text, String, Text, Table, Index, ForeignKey,Integer,Float
from typing import dict, Any
from datetime import datetime
import uuid 
from sqlalchemy.orm import foreign, remote
from sqlalchemy import Uuid as UUID


convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Create metadata with naming convention
metadata = MetaData(naming_convention=convention)

# Create base class with common fields
class Base(DeclarativeBase):
    metadata = metadata
    
    __abstract__ = True
    
    id = Column(UUID(), primary_key=True, default=uuid.uuid4)
    date_created = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    date_modified = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True)

# Event listeners for automatic timestamp updates
@event.listens_for(Base, "before_insert", propagate=True)
def set_date_created(mapper, connection, target):
    target.date_created = datetime.now()
    if target.date_modified is None:
        target.date_modified = datetime.now()

@event.listens_for(Base, "before_update", propagate=True)
def set_date_modified(mapper, connection, target):
    target.date_modified = datetime.now()


# Association tables with indexes for faster joins
paper_authors = Table(
    'paper_authors',
    Base.metadata,
    Column('paper_id', UUID(), ForeignKey('papers.id'), nullable=False),
    Column('author_id', UUID(), ForeignKey('authors.id'), nullable=False),
    Index('ix_paper_authors_paper_id', 'paper_id'),  
    Index('ix_paper_authors_author_id', 'author_id')  
)

paper_categories = Table(
    'paper_categories',
    Base.metadata,
    Column('paper_id', UUID(), ForeignKey('papers.id'), nullable=False),
    Column('category_id', UUID(), ForeignKey('categories.id'), nullable=False),
    Index('ix_paper_categories_paper_id', 'paper_id'),
    Index('ix_paper_categories_category_id', 'category_id')
)

paper_references = Table(
    'paper_references',
    Base.metadata,
    Column('citing_paper_id', UUID(), ForeignKey('papers.id'), nullable=False),
    Column('cited_paper_id', UUID(), ForeignKey('papers.id'), nullable=False),
    Index('ix_paper_references_citing_id', 'citing_paper_id'),
    Index('ix_paper_references_cited_id', 'cited_paper_id')
)

class Paper(Base):
    __tablename__ = 'papers'

    arxiv_id = Column(String(20), unique=True, nullable=False, index=True)
    title = Column(String(512), nullable=False)
    summary = Column(Text, nullable=False)
    content = Column(Text) 

    source = Column(String(255))
    comment = Column(Text)
    journal_ref = Column(String(255))
    primary_category = Column(String(20), nullable=False, index=True)
    

    total_tokens = Column(Integer)
    chunk_count = Column(Integer)
    
    published = Column(String(8), nullable=False, index=True)
    updated = Column(String(8), nullable=False, index=True)

    authors = relationship(
        "Author",
        secondary=paper_authors,
        back_populates="papers",
        lazy='joined'
    )
    
    categories = relationship(
        "Category",
        secondary=paper_categories,
        back_populates="papers",
        lazy='joined'
    )

    citations = relationship(
        'Paper',
        secondary=paper_references,
        primaryjoin="Paper.id == paper_references.c.citing_paper_id",
        secondaryjoin="Paper.id == paper_references.c.cited_paper_id",
        backref='cited_by'
    )

    __table_args__ = (
        Index('ix_papers_published_primary_category', 'published', 'primary_category'),
        Index('ix_papers_title', 'title'),
    )

    def add_citation(self, cited_paper_arxiv_id: str, session: Session) -> None:
        """Helper method to add citation by arxiv_id"""
        cited_paper = session.query(Paper).filter_by(arxiv_id=cited_paper_arxiv_id).first()
        if cited_paper:
            self.citations.append(cited_paper)

    def to_dict(self, include_citations: bool = True) -> dict[str, Any]:
        """
        Convert paper to dictionary representation.
        Args:
            include_citations: Whether to include citation data
            include_chunks: Whether to include chunk data
        """
        data = {
            "id": str(self.id),
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "source": self.source,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "primary_category": self.primary_category,
            "published": self.published,
            "updated": self.updated,
            "total_tokens": self.total_tokens,
            "chunk_count": self.chunk_count,
            "authors": [author.to_dict() for author in self.authors],
            "categories": [category.to_dict() for category in self.categories],
            "date_created": self.date_created.isoformat() if self.date_created else None,
            "date_modified": self.date_modified.isoformat() if self.date_modified else None
        }
        
        if include_citations:
            data.update({
                "citations": [cited.arxiv_id for cited in self.citations],
                "cited_by": [citing.arxiv_id for citing in self.cited_by]
            })
        
        return data

class Author(Base):

    __tablename__ = 'authors'

    name = Column(String(255), nullable=False, index=True)  # Indexed for name searches
    
    email = Column(String(255), unique=True, index=True)  
    institution = Column(String(255), index=True)        
    
    papers = relationship(
        "Paper",
        secondary=paper_authors,
        back_populates="authors",
        lazy='select'
    )

    __table_args__ = (
        Index('ix_authors_name_institution', 'name', 'institution'),  # Composite index for name+institution searches
    )

    def to_dict(self, include_papers: bool = False) -> dict[str, Any]:
        """
        Convert author to dictionary representation.
        Args:
            include_papers: Whether to include paper data
        """
        data = {
            "id": str(self.id),
            "name": self.name,
            "email": self.email,
            "institution": self.institution,
            "date_created": self.date_created.isoformat() if self.date_created else None,
            "date_modified": self.date_modified.isoformat() if self.date_modified else None
        }
        
        if include_papers:
            data["papers"] = [
                {"id": str(paper.id), "title": paper.title, "arxiv_id": paper.arxiv_id}
                for paper in self.papers
            ]
        
        return data

class Category(Base):

    __tablename__ = 'categories'

    code = Column(String(20), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)

    papers = relationship(
        "Paper",
        secondary=paper_categories,
        back_populates="categories",
        lazy='select'
    )

    __table_args__ = (
        Index('ix_categories_code_name', 'code', 'name'),  # Composite index for code+name searches
    )

    def to_dict(self, include_papers: bool = False) -> dict[str, Any]:
        """
        Convert category to dictionary representation.
        Args:
            include_papers: Whether to include paper data
        """
        data = {
            "id": str(self.id),
            "code": self.code,
            "name": self.name,
            "description": self.description,
            "date_created": self.date_created.isoformat() if self.date_created else None,
            "date_modified": self.date_modified.isoformat() if self.date_modified else None
        }
        
        if include_papers:
            data["papers"] = [
                {"id": str(paper.id), "title": paper.title, "arxiv_id": paper.arxiv_id}
                for paper in self.papers
            ]
        
        return data
    