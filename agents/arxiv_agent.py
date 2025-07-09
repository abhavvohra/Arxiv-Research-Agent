import arxiv
from typing import List, Dict, Any
from datetime import datetime

class ArxivAgent:
    """Enhanced ArXiv API integration agent for comprehensive paper retrieval"""
    
    def __init__(self, max_papers: int = 10):
        self.max_papers = max_papers
        self.client = arxiv.Client()
    
    def search_papers(self, query: str, categories: List[str] = None) -> List[Dict[str, Any]]:
        """Search ArXiv for papers with enhanced filtering and metadata"""
        
        # Enhance query with category filters if provided
        if categories:
            category_query = " OR ".join([f"cat:{cat}" for cat in categories])
            enhanced_query = f"({query}) AND ({category_query})"
        else:
            enhanced_query = query
        
        search = arxiv.Search(
            query=enhanced_query,
            max_results=self.max_papers,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in self.client.results(search):
            # Extract additional metadata
            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "url": result.entry_id,
                "published": result.published.strftime("%Y-%m-%d"),
                "updated": result.updated.strftime("%Y-%m-%d") if result.updated else None,
                "categories": result.categories,
                "primary_category": result.primary_category,
                "pdf_url": result.pdf_url,
                "doi": result.doi,
                "journal_ref": result.journal_ref,
                "comment": result.comment,
                "links": [link.href for link in result.links],
                "citation_count": 0,  # Would need external API for actual count
                "retrieved_at": datetime.now().isoformat()
            }
            papers.append(paper)
        
        return papers
    
    def search_by_author(self, author_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search papers by specific author"""
        query = f"au:{author_name}"
        return self.search_papers(query)
    
    def search_recent_papers(self, query: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """Search for recent papers within specified timeframe"""
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}0000 TO {end_date.strftime('%Y%m%d')}2359]"
        enhanced_query = f"({query}) AND {date_query}"
        
        search = arxiv.Search(
            query=enhanced_query,
            max_results=self.max_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in self.client.results(search):
            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "url": result.entry_id,
                "published": result.published.strftime("%Y-%m-%d"),
                "categories": result.categories,
                "pdf_url": result.pdf_url,
                "retrieved_at": datetime.now().isoformat()
            }
            papers.append(paper)
        
        return papers
    
    def get_paper_details(self, arxiv_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific paper by ArXiv ID"""
        search = arxiv.Search(id_list=[arxiv_id])
        
        try:
            paper = next(self.client.results(search))
            return {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "url": paper.entry_id,
                "published": paper.published.strftime("%Y-%m-%d"),
                "updated": paper.updated.strftime("%Y-%m-%d") if paper.updated else None,
                "categories": paper.categories,
                "primary_category": paper.primary_category,
                "pdf_url": paper.pdf_url,
                "doi": paper.doi,
                "journal_ref": paper.journal_ref,
                "comment": paper.comment,
                "links": [link.href for link in paper.links],
                "retrieved_at": datetime.now().isoformat()
            }
        except StopIteration:
            return None
    
    def get_category_papers(self, category: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Get papers from a specific ArXiv category"""
        query = f"cat:{category}"
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in self.client.results(search):
            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "url": result.entry_id,
                "published": result.published.strftime("%Y-%m-%d"),
                "categories": result.categories,
                "pdf_url": result.pdf_url,
                "retrieved_at": datetime.now().isoformat()
            }
            papers.append(paper)
        
        return papers