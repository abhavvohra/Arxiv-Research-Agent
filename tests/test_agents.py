import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

# Import agents
from agents.arxiv_agent import ArxivAgent
from agents.vector_agent import VectorDatabaseAgent
from agents.analysis_agent import AnalysisAgent
from agents.synthesis_agent import SynthesisAgent
from agents.reflection_agent import ReflectionAgent
from agents.report_generator import ReportGenerator
from agents.theme_extractor import ThemeExtractor

class TestArxivAgent:
    """Test cases for ArxivAgent"""
    
    def test_init(self):
        agent = ArxivAgent(max_papers=5)
        assert agent.max_papers == 5
        assert agent.client is not None
    
    @patch('arxiv.Client')
    def test_search_papers(self, mock_client):
        # Mock arxiv search result
        mock_result = Mock()
        mock_result.title = "Test Paper"
        mock_result.authors = [Mock(name="John Doe")]
        mock_result.summary = "Test abstract"
        mock_result.entry_id = "http://arxiv.org/abs/1234.5678"
        mock_result.published = datetime(2023, 1, 1)
        mock_result.categories = ["cs.AI"]
        mock_result.pdf_url = "http://arxiv.org/pdf/1234.5678"
        mock_result.primary_category = "cs.AI"
        mock_result.updated = None
        mock_result.doi = None
        mock_result.journal_ref = None
        mock_result.comment = None
        mock_result.links = []
        
        mock_client.return_value.results.return_value = [mock_result]
        
        agent = ArxivAgent(max_papers=1)
        papers = agent.search_papers("machine learning")
        
        assert len(papers) == 1
        assert papers[0]["title"] == "Test Paper"
        assert papers[0]["authors"] == ["John Doe"]
    
    def test_generate_paper_id(self):
        agent = ArxivAgent()
        paper = {
            "title": "Test Paper",
            "url": "http://arxiv.org/abs/1234.5678"
        }
        
        paper_id = agent._generate_paper_id(paper)
        assert isinstance(paper_id, str)
        assert len(paper_id) == 32  # MD5 hash length

class TestVectorDatabaseAgent:
    """Test cases for VectorDatabaseAgent"""
    
    def test_init(self):
        agent = VectorDatabaseAgent(collection_name="test_collection")
        assert agent.collection_name == "test_collection"
        assert agent.collection is not None
    
    def test_create_document_text(self):
        agent = VectorDatabaseAgent()
        paper = {
            "title": "Test Paper",
            "authors": ["John Doe", "Jane Smith"],
            "summary": "This is a test abstract",
            "categories": ["cs.AI", "cs.LG"]
        }
        
        doc_text = agent._create_document_text(paper)
        
        assert "Test Paper" in doc_text
        assert "John Doe" in doc_text
        assert "This is a test abstract" in doc_text
        assert "cs.AI" in doc_text
    
    def test_create_metadata(self):
        agent = VectorDatabaseAgent()
        paper = {
            "title": "Test Paper",
            "authors": ["John Doe"],
            "url": "http://arxiv.org/abs/1234.5678",
            "published": "2023-01-01",
            "categories": ["cs.AI"]
        }
        
        metadata = agent._create_metadata(paper)
        
        assert metadata["title"] == "Test Paper"
        assert metadata["authors"] == "John Doe"
        assert metadata["url"] == "http://arxiv.org/abs/1234.5678"
        assert "added_at" in metadata

class TestAnalysisAgent:
    """Test cases for AnalysisAgent"""
    
    def test_init(self):
        agent = AnalysisAgent()
        assert agent.llm is not None
        assert agent.tokenizer is not None
    
    def test_truncate_text(self):
        agent = AnalysisAgent()
        
        # Test short text
        short_text = "This is a short text."
        result = agent._truncate_text(short_text, 100)
        assert result == short_text
        
        # Test long text
        long_text = "This is a very long text. " * 50
        result = agent._truncate_text(long_text, 100)
        assert len(result) <= 103  # 100 + "..."
    
    def test_parse_analysis_response(self):
        agent = AnalysisAgent()
        
        # Test valid JSON response
        json_response = '{"analysis": "test", "relevance_score": 8.0}'
        result = agent._parse_analysis_response(json_response)
        
        assert result["analysis"] == "test"
        assert result["relevance_score"] == 8.0
        
        # Test invalid JSON response
        invalid_response = "This is not JSON"
        result = agent._parse_analysis_response(invalid_response)
        
        assert result["analysis"] == invalid_response
        assert "relevance_score" in result

class TestSynthesisAgent:
    """Test cases for SynthesisAgent"""
    
    def test_init(self):
        agent = SynthesisAgent()
        assert agent.llm is not None
    
    def test_prepare_paper_summaries(self):
        agent = SynthesisAgent()
        
        analyzed_papers = [
            {
                "paper_info": {
                    "title": "Test Paper 1",
                    "authors": ["John Doe"],
                    "published": "2023-01-01"
                },
                "analysis": "This is a test analysis.",
                "structured_insights": {
                    "main_contribution": "Test contribution"
                },
                "relevance_score": 8.0,
                "quality_score": 7.5
            }
        ]
        
        summaries = agent._prepare_paper_summaries(analyzed_papers)
        
        assert "Test Paper 1" in summaries
        assert "John Doe" in summaries
        assert "Test contribution" in summaries
        assert "8.0" in summaries

class TestReflectionAgent:
    """Test cases for ReflectionAgent"""
    
    def test_init(self):
        agent = ReflectionAgent()
        assert agent.llm is not None
        assert agent.quality_criteria is not None
    
    def test_create_default_evaluation(self):
        agent = ReflectionAgent()
        evaluation = agent._create_default_evaluation()
        
        assert "completeness" in evaluation
        assert "accuracy" in evaluation
        assert "clarity" in evaluation
        assert "depth" in evaluation
        assert "actionability" in evaluation
        
        for criterion in evaluation.values():
            assert "score" in criterion
            assert "observations" in criterion
            assert "improvements" in criterion
    
    def test_calculate_confidence_score(self):
        agent = ReflectionAgent()
        
        evaluation = {
            "accuracy": {"score": 8},
            "completeness": {"score": 7},
            "depth": {"score": 6},
            "clarity": {"score": 9},
            "actionability": {"score": 5}
        }
        
        confidence = agent._calculate_confidence_score(evaluation)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.6  # Should be reasonable given the scores

class TestReportGenerator:
    """Test cases for ReportGenerator"""
    
    def test_init(self):
        agent = ReportGenerator()
        assert agent.llm is not None
        assert agent.citation_style == "academic"
    
    def test_format_authors(self):
        agent = ReportGenerator()
        
        # Test single author
        authors = ["John Doe"]
        result = agent._format_authors(authors)
        assert result == "John Doe"
        
        # Test two authors
        authors = ["John Doe", "Jane Smith"]
        result = agent._format_authors(authors)
        assert result == "John Doe and Jane Smith"
        
        # Test multiple authors
        authors = ["John Doe", "Jane Smith", "Bob Johnson"]
        result = agent._format_authors(authors)
        assert result == "John Doe, Jane Smith, and Bob Johnson"
        
        # Test many authors
        authors = ["A", "B", "C", "D", "E", "F"]
        result = agent._format_authors(authors)
        assert result == "A et al."
    
    def test_extract_arxiv_id(self):
        agent = ReportGenerator()
        
        url = "http://arxiv.org/abs/1234.5678"
        arxiv_id = agent._extract_arxiv_id(url)
        assert arxiv_id == "1234.5678"
        
        # Test invalid URL
        invalid_url = "http://example.com/paper"
        arxiv_id = agent._extract_arxiv_id(invalid_url)
        assert arxiv_id == ""

class TestThemeExtractor:
    """Test cases for ThemeExtractor"""
    
    def test_init(self):
        agent = ThemeExtractor()
        assert agent.llm is not None
    
    def test_prepare_content_for_analysis(self):
        agent = ThemeExtractor()
        
        analyzed_papers = [
            {
                "paper_info": {
                    "title": "Test Paper",
                    "categories": ["cs.AI"]
                },
                "structured_insights": {
                    "main_contribution": "Test contribution",
                    "methodology": "Test method"
                },
                "analysis": "This is a test analysis."
            }
        ]
        
        content = agent._prepare_content_for_analysis(analyzed_papers)
        
        assert "Test Paper" in content
        assert "cs.AI" in content
        assert "Test contribution" in content
        assert "Test method" in content
    
    def test_extract_keyword_themes(self):
        agent = ThemeExtractor()
        
        analyzed_papers = [
            {
                "paper_info": {
                    "title": "Machine Learning Applications",
                    "summary": "This paper discusses machine learning applications in healthcare"
                },
                "analysis": "The paper presents novel machine learning algorithms for medical diagnosis"
            }
        ]
        
        result = agent.extract_keyword_themes(analyzed_papers)
        
        assert "keyword_themes" in result
        assert "word_frequencies" in result
        assert isinstance(result["keyword_themes"], list)
        assert isinstance(result["word_frequencies"], dict)

# Integration tests
class TestIntegration:
    """Integration tests for agent interactions"""
    
    def test_paper_analysis_workflow(self):
        # Create sample paper
        paper = {
            "title": "Test Machine Learning Paper",
            "authors": ["John Doe", "Jane Smith"],
            "summary": "This paper presents a novel approach to machine learning in healthcare applications.",
            "url": "http://arxiv.org/abs/1234.5678",
            "published": "2023-01-01",
            "categories": ["cs.AI", "cs.LG"]
        }
        
        # Test analysis agent
        analysis_agent = AnalysisAgent()
        # Skip actual LLM call for testing
        analysis_result = {
            "paper_info": paper,
            "analysis": "Test analysis",
            "relevance_score": 8.0,
            "quality_score": 7.5
        }
        
        # Test theme extraction
        theme_extractor = ThemeExtractor()
        # Skip actual LLM call for testing
        themes = {
            "themes": ["machine learning", "healthcare"],
            "gaps": ["more data needed"],
            "contradictions": []
        }
        
        assert analysis_result["relevance_score"] == 8.0
        assert themes["themes"] == ["machine learning", "healthcare"]

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])