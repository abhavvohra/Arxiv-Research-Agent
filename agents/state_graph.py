from typing import Dict, List, Any, Optional, Annotated
from dataclasses import dataclass, field
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

@dataclass
class ResearchState:
    """Enhanced state object for the research workflow with comprehensive tracking"""
    
    # Core research data
    query: str
    papers: List[Dict[str, Any]] = field(default_factory=list)
    analyzed_papers: List[Dict[str, Any]] = field(default_factory=list)
    synthesis: str = ""
    report: str = ""
    
    # Reflection and iteration
    reflection_feedback: str = ""
    iteration_count: int = 0
    max_iterations: int = 3
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # Vector database and semantic search
    vector_db_results: List[Dict[str, Any]] = field(default_factory=list)
    semantic_clusters: List[Dict[str, Any]] = field(default_factory=list)
    
    # Research metadata
    research_start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    research_end_time: Optional[str] = None
    total_processing_time: Optional[float] = None
    
    # Quality metrics
    relevance_scores: List[float] = field(default_factory=list)
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    
    # Research configuration
    search_categories: List[str] = field(default_factory=list)
    max_papers: int = 10
    include_recent_only: bool = False
    days_back: int = 30
    
    # Conversation history for agent interactions
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Research insights
    key_themes: List[str] = field(default_factory=list)
    research_gaps: List[str] = field(default_factory=list)
    future_directions: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)

class ResearchWorkflow:
    """Enhanced StateGraph orchestration framework for research workflow"""
    
    def __init__(self):
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the enhanced StateGraph workflow with comprehensive nodes and edges"""
        
        workflow = StateGraph(ResearchState)
        
        # Add all workflow nodes
        workflow.add_node("initialize", self._initialize_research)
        workflow.add_node("retrieve_papers", self._retrieve_papers)
        workflow.add_node("quality_filter", self._quality_filter)
        workflow.add_node("vector_search", self._vector_search)
        workflow.add_node("semantic_clustering", self._semantic_clustering)
        workflow.add_node("analyze_papers", self._analyze_papers)
        workflow.add_node("extract_themes", self._extract_themes)
        workflow.add_node("synthesize", self._synthesize)
        workflow.add_node("reflect", self._reflect)
        workflow.add_node("improve_synthesis", self._improve_synthesis)
        workflow.add_node("generate_report", self._generate_report)
        workflow.add_node("finalize", self._finalize_research)
        
        # Define workflow edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "retrieve_papers")
        workflow.add_edge("retrieve_papers", "quality_filter")
        workflow.add_edge("quality_filter", "vector_search")
        workflow.add_edge("vector_search", "semantic_clustering")
        workflow.add_edge("semantic_clustering", "analyze_papers")
        workflow.add_edge("analyze_papers", "extract_themes")
        workflow.add_edge("extract_themes", "synthesize")
        workflow.add_edge("synthesize", "reflect")
        
        # Conditional edge for iteration
        workflow.add_conditional_edges(
            "reflect",
            self._should_iterate,
            {
                "iterate": "improve_synthesis",
                "finish": "generate_report"
            }
        )
        
        workflow.add_edge("improve_synthesis", "synthesize")
        workflow.add_edge("generate_report", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _initialize_research(self, state: ResearchState) -> Dict[str, Any]:
        """Initialize research session with metadata and validation"""
        state.research_start_time = datetime.now().isoformat()
        
        # Validate query
        if not state.query or len(state.query.strip()) < 3:
            state.errors.append("Query must be at least 3 characters long")
            return {"state": state}
        
        # Set default configuration if not provided
        if not state.search_categories:
            state.search_categories = ["cs.AI", "cs.LG", "cs.CL", "stat.ML"]
        
        return {"state": state}
    
    def _retrieve_papers(self, state: ResearchState) -> Dict[str, Any]:
        """Retrieve papers from ArXiv with enhanced filtering"""
        from agents.arxiv_agent import ArxivAgent
        
        try:
            arxiv_agent = ArxivAgent(max_papers=state.max_papers)
            
            if state.include_recent_only:
                papers = arxiv_agent.search_recent_papers(
                    state.query, 
                    days_back=state.days_back
                )
            else:
                papers = arxiv_agent.search_papers(
                    state.query,
                    categories=state.search_categories
                )
            
            state.papers = papers
            
            if not papers:
                state.warnings.append("No papers found for the given query")
            
        except Exception as e:
            state.errors.append(f"Error retrieving papers: {str(e)}")
        
        return {"state": state}
    
    def _quality_filter(self, state: ResearchState) -> Dict[str, Any]:
        """Filter papers based on quality metrics"""
        if not state.papers:
            return {"state": state}
        
        # Simple quality filtering based on abstract length and recency
        filtered_papers = []
        for paper in state.papers:
            quality_score = 0
            
            # Check abstract length (longer abstracts often indicate more detailed work)
            if len(paper.get("summary", "")) > 150:
                quality_score += 1
            
            # Check if paper has DOI (often indicates peer review)
            if paper.get("doi"):
                quality_score += 1
            
            # Check if paper has journal reference
            if paper.get("journal_ref"):
                quality_score += 1
            
            # Prefer papers with more categories (interdisciplinary)
            if len(paper.get("categories", [])) > 1:
                quality_score += 0.5
            
            paper["quality_score"] = quality_score
            
            # Keep papers with quality score >= 1
            if quality_score >= 1:
                filtered_papers.append(paper)
        
        state.papers = filtered_papers[:state.max_papers]  # Limit to max papers
        
        return {"state": state}
    
    def _vector_search(self, state: ResearchState) -> Dict[str, Any]:
        """Add papers to vector DB and perform semantic search"""
        from agents.vector_agent import VectorDatabaseAgent
        
        try:
            vector_agent = VectorDatabaseAgent()
            
            # Add papers to vector database
            vector_agent.add_papers(state.papers)
            
            # Perform semantic search
            vector_results = vector_agent.semantic_search(
                state.query, 
                n_results=min(len(state.papers), 10)
            )
            
            state.vector_db_results = vector_results
            
        except Exception as e:
            state.errors.append(f"Error in vector search: {str(e)}")
        
        return {"state": state}
    
    def _semantic_clustering(self, state: ResearchState) -> Dict[str, Any]:
        """Perform semantic clustering of papers"""
        # This would implement clustering logic
        # For now, we'll create simple thematic groups
        
        clusters = {}
        for paper in state.papers:
            primary_category = paper.get("primary_category", "unknown")
            if primary_category not in clusters:
                clusters[primary_category] = []
            clusters[primary_category].append(paper)
        
        state.semantic_clusters = [
            {"category": cat, "papers": papers, "count": len(papers)}
            for cat, papers in clusters.items()
        ]
        
        return {"state": state}
    
    def _analyze_papers(self, state: ResearchState) -> Dict[str, Any]:
        """Analyze individual papers with enhanced insights"""
        from agents.analysis_agent import AnalysisAgent
        
        try:
            analysis_agent = AnalysisAgent()
            analyzed_papers = []
            relevance_scores = []
            
            for paper in state.papers:
                analysis = analysis_agent.analyze_paper(paper)
                analyzed_papers.append(analysis)
                
                # Extract relevance score from analysis
                relevance_score = analysis.get("relevance_score", 5.0)
                relevance_scores.append(relevance_score)
            
            state.analyzed_papers = analyzed_papers
            state.relevance_scores = relevance_scores
            
        except Exception as e:
            state.errors.append(f"Error analyzing papers: {str(e)}")
        
        return {"state": state}
    
    def _extract_themes(self, state: ResearchState) -> Dict[str, Any]:
        """Extract key themes from analyzed papers"""
        from agents.theme_extractor import ThemeExtractor
        
        try:
            theme_extractor = ThemeExtractor()
            themes = theme_extractor.extract_themes(state.analyzed_papers)
            
            state.key_themes = themes.get("themes", [])
            state.research_gaps = themes.get("gaps", [])
            state.contradictions = themes.get("contradictions", [])
            
        except Exception as e:
            state.errors.append(f"Error extracting themes: {str(e)}")
        
        return {"state": state}
    
    def _synthesize(self, state: ResearchState) -> Dict[str, Any]:
        """Synthesize findings from analyzed papers"""
        from agents.synthesis_agent import SynthesisAgent
        
        try:
            synthesis_agent = SynthesisAgent()
            synthesis = synthesis_agent.synthesize_findings(
                state.analyzed_papers, 
                state.query,
                themes=state.key_themes,
                gaps=state.research_gaps
            )
            
            state.synthesis = synthesis
            
        except Exception as e:
            state.errors.append(f"Error in synthesis: {str(e)}")
        
        return {"state": state}
    
    def _reflect(self, state: ResearchState) -> Dict[str, Any]:
        """Reflect on synthesis and provide improvement feedback"""
        from agents.reflection_agent import ReflectionAgent
        
        try:
            reflection_agent = ReflectionAgent()
            reflection = reflection_agent.reflect_on_synthesis(
                state.synthesis, 
                state.query,
                iteration=state.iteration_count
            )
            
            state.reflection_feedback = reflection.get("feedback", "")
            state.improvement_suggestions = reflection.get("suggestions", [])
            state.confidence_score = reflection.get("confidence_score", 0.0)
            state.completeness_score = reflection.get("completeness_score", 0.0)
            
            state.iteration_count += 1
            
        except Exception as e:
            state.errors.append(f"Error in reflection: {str(e)}")
        
        return {"state": state}
    
    def _should_iterate(self, state: ResearchState) -> str:
        """Enhanced decision logic for iteration"""
        # Don't iterate if we have errors
        if state.errors:
            return "finish"
        
        # Don't iterate if we've reached max iterations
        if state.iteration_count >= state.max_iterations:
            return "finish"
        
        # Iterate if confidence or completeness is low
        if (state.confidence_score < 0.7 or state.completeness_score < 0.7) and state.iteration_count < state.max_iterations:
            return "iterate"
        
        return "finish"
    
    def _improve_synthesis(self, state: ResearchState) -> Dict[str, Any]:
        """Improve synthesis based on reflection feedback"""
        from agents.synthesis_agent import SynthesisAgent
        
        try:
            synthesis_agent = SynthesisAgent()
            improved_synthesis = synthesis_agent.improve_synthesis(
                state.synthesis,
                state.reflection_feedback,
                state.improvement_suggestions
            )
            
            state.synthesis = improved_synthesis
            
        except Exception as e:
            state.errors.append(f"Error improving synthesis: {str(e)}")
        
        return {"state": state}
    
    def _generate_report(self, state: ResearchState) -> Dict[str, Any]:
        """Generate comprehensive research report"""
        from agents.report_generator import ReportGenerator
        
        try:
            report_generator = ReportGenerator()
            report = report_generator.generate_comprehensive_report(
                query=state.query,
                synthesis=state.synthesis,
                papers=state.papers,
                themes=state.key_themes,
                gaps=state.research_gaps,
                contradictions=state.contradictions,
                clusters=state.semantic_clusters,
                metadata={
                    "papers_analyzed": len(state.papers),
                    "iterations": state.iteration_count,
                    "confidence_score": state.confidence_score,
                    "completeness_score": state.completeness_score
                }
            )
            
            state.report = report
            
        except Exception as e:
            state.errors.append(f"Error generating report: {str(e)}")
        
        return {"state": state}
    
    def _finalize_research(self, state: ResearchState) -> Dict[str, Any]:
        """Finalize research with metadata and cleanup"""
        state.research_end_time = datetime.now().isoformat()
        
        # Calculate processing time
        start_time = datetime.fromisoformat(state.research_start_time)
        end_time = datetime.fromisoformat(state.research_end_time)
        state.total_processing_time = (end_time - start_time).total_seconds()
        
        return {"state": state}
    
    async def run_research(self, initial_state: ResearchState) -> ResearchState:
        """Run the complete research workflow"""
        try:
            final_state = await self.graph.ainvoke(initial_state)
            return final_state
        except Exception as e:
            initial_state.errors.append(f"Workflow error: {str(e)}")
            return initial_state