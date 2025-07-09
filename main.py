#!/usr/bin/env python3
"""
Automated Academic Research System using LangGraph's Multi-Agent Architecture
"""

import os
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
import arxiv
import chromadb
from chromadb.utils import embedding_functions
import tiktoken
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ResearchState:
    """State object for the research workflow"""
    query: str
    papers: List[Dict[str, Any]] = field(default_factory=list)
    analyzed_papers: List[Dict[str, Any]] = field(default_factory=list)
    synthesis: str = ""
    report: str = ""
    reflection_feedback: str = ""
    iteration_count: int = 0
    max_iterations: int = 3
    vector_db_results: List[Dict[str, Any]] = field(default_factory=list)

class ArxivAgent:
    """Agent for retrieving papers from ArXiv API"""
    
    def __init__(self, max_papers: int = 10):
        self.max_papers = max_papers
        self.client = arxiv.Client()
    
    def search_papers(self, query: str) -> List[Dict[str, Any]]:
        """Search ArXiv for papers matching the query"""
        search = arxiv.Search(
            query=query,
            max_results=self.max_papers,
            sort_by=arxiv.SortCriterion.Relevance,
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
                "pdf_url": result.pdf_url
            }
            papers.append(paper)
        
        return papers

class VectorDatabaseAgent:
    """Agent for semantic search using vector database"""
    
    def __init__(self, collection_name: str = "research_papers"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def add_papers(self, papers: List[Dict[str, Any]]):
        """Add papers to the vector database"""
        documents = []
        metadatas = []
        ids = []
        
        for i, paper in enumerate(papers):
            doc_text = f"Title: {paper['title']}\nAbstract: {paper['summary']}\nAuthors: {', '.join(paper['authors'])}"
            documents.append(doc_text)
            metadatas.append({
                "title": paper["title"],
                "authors": ", ".join(paper["authors"]),
                "url": paper["url"],
                "published": paper["published"]
            })
            ids.append(f"paper_{i}_{hash(paper['title'])}")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on the vector database"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        search_results = []
        for i, doc in enumerate(results["documents"][0]):
            search_results.append({
                "document": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return search_results

class AnalysisAgent:
    """Agent for analyzing individual papers"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    def analyze_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single paper and extract key insights"""
        prompt = f"""
        Analyze this research paper and provide a structured analysis:
        
        Title: {paper['title']}
        Authors: {', '.join(paper['authors'])}
        Abstract: {paper['summary']}
        Published: {paper['published']}
        
        Provide analysis in the following structure:
        1. Main Contribution
        2. Key Findings
        3. Methodology
        4. Strengths
        5. Limitations
        6. Relevance Score (1-10)
        
        Keep the analysis concise but comprehensive.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "paper_info": paper,
            "analysis": response.content,
            "analyzed_at": datetime.now().isoformat()
        }

class SynthesisAgent:
    """Agent for synthesizing insights from multiple papers"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
    
    def synthesize_findings(self, analyzed_papers: List[Dict[str, Any]], query: str) -> str:
        """Synthesize findings from multiple analyzed papers"""
        papers_text = "\n\n".join([
            f"Paper {i+1}:\nTitle: {paper['paper_info']['title']}\nAnalysis: {paper['analysis']}"
            for i, paper in enumerate(analyzed_papers)
        ])
        
        prompt = f"""
        Based on the following analyzed research papers related to "{query}", 
        provide a comprehensive synthesis that includes:
        
        1. Common themes and patterns across papers
        2. Contradictions or debates in the field
        3. Research gaps identified
        4. Future research directions
        5. Practical implications
        
        Analyzed Papers:
        {papers_text}
        
        Provide a well-structured synthesis that connects insights across papers.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

class ReflectionAgent:
    """Agent for iterative improvement through reflection"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
    
    def reflect_on_synthesis(self, synthesis: str, query: str) -> str:
        """Provide reflection and improvement suggestions"""
        prompt = f"""
        Review the following research synthesis for the query "{query}" and provide constructive feedback:
        
        Synthesis:
        {synthesis}
        
        Evaluate the synthesis on:
        1. Completeness - Are all key aspects covered?
        2. Accuracy - Are the connections and insights accurate?
        3. Clarity - Is the synthesis clear and well-organized?
        4. Depth - Does it provide sufficient depth of analysis?
        5. Actionability - Are the insights actionable for researchers?
        
        Provide specific suggestions for improvement.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

class ReportGenerator:
    """Agent for generating structured research reports"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
    
    def generate_report(self, query: str, synthesis: str, papers: List[Dict[str, Any]]) -> str:
        """Generate a structured research report with proper citations"""
        citations = []
        for i, paper in enumerate(papers, 1):
            authors = ", ".join(paper["authors"][:3])  # First 3 authors
            if len(paper["authors"]) > 3:
                authors += " et al."
            citations.append(f"[{i}] {authors}. \"{paper['title']}\". arXiv preprint. {paper['published']}. {paper['url']}")
        
        citations_text = "\n".join(citations)
        
        prompt = f"""
        Generate a comprehensive research report based on the synthesis and analyzed papers.
        
        Query: {query}
        Synthesis: {synthesis}
        
        Structure the report as follows:
        1. Executive Summary
        2. Introduction
        3. Literature Review
        4. Key Findings
        5. Analysis and Discussion
        6. Research Gaps and Future Directions
        7. Conclusions
        8. References
        
        Use proper academic formatting and include citations [1], [2], etc. where appropriate.
        Make the report publication-ready and professional.
        
        Available Citations:
        {citations_text}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

class ResearchOrchestrator:
    """Main orchestrator using LangGraph StateGraph"""
    
    def __init__(self):
        self.arxiv_agent = ArxivAgent()
        self.vector_db_agent = VectorDatabaseAgent()
        self.analysis_agent = AnalysisAgent()
        self.synthesis_agent = SynthesisAgent()
        self.reflection_agent = ReflectionAgent()
        self.report_generator = ReportGenerator()
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the StateGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("retrieve_papers", self._retrieve_papers)
        workflow.add_node("vector_search", self._vector_search)
        workflow.add_node("analyze_papers", self._analyze_papers)
        workflow.add_node("synthesize", self._synthesize)
        workflow.add_node("reflect", self._reflect)
        workflow.add_node("generate_report", self._generate_report)
        
        # Add edges
        workflow.set_entry_point("retrieve_papers")
        workflow.add_edge("retrieve_papers", "vector_search")
        workflow.add_edge("vector_search", "analyze_papers")
        workflow.add_edge("analyze_papers", "synthesize")
        workflow.add_edge("synthesize", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            self._should_iterate,
            {
                "iterate": "synthesize",
                "finish": "generate_report"
            }
        )
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    def _retrieve_papers(self, state: ResearchState) -> ResearchState:
        """Retrieve papers from ArXiv"""
        papers = self.arxiv_agent.search_papers(state.query)
        state.papers = papers
        return state
    
    def _vector_search(self, state: ResearchState) -> ResearchState:
        """Add papers to vector DB and perform semantic search"""
        self.vector_db_agent.add_papers(state.papers)
        vector_results = self.vector_db_agent.semantic_search(state.query)
        state.vector_db_results = vector_results
        return state
    
    def _analyze_papers(self, state: ResearchState) -> ResearchState:
        """Analyze individual papers"""
        analyzed_papers = []
        for paper in state.papers:
            analysis = self.analysis_agent.analyze_paper(paper)
            analyzed_papers.append(analysis)
        
        state.analyzed_papers = analyzed_papers
        return state
    
    def _synthesize(self, state: ResearchState) -> ResearchState:
        """Synthesize findings from analyzed papers"""
        synthesis = self.synthesis_agent.synthesize_findings(
            state.analyzed_papers, state.query
        )
        state.synthesis = synthesis
        return state
    
    def _reflect(self, state: ResearchState) -> ResearchState:
        """Reflect on synthesis and provide feedback"""
        reflection = self.reflection_agent.reflect_on_synthesis(
            state.synthesis, state.query
        )
        state.reflection_feedback = reflection
        state.iteration_count += 1
        return state
    
    def _should_iterate(self, state: ResearchState) -> str:
        """Determine if we should iterate or finish"""
        if state.iteration_count < state.max_iterations:
            return "iterate"
        return "finish"
    
    def _generate_report(self, state: ResearchState) -> ResearchState:
        """Generate final research report"""
        report = self.report_generator.generate_report(
            state.query, state.synthesis, state.papers
        )
        state.report = report
        return state
    
    async def run_research(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Run the complete research workflow"""
        initial_state = ResearchState(
            query=query,
            max_iterations=max_iterations
        )
        
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "query": final_state.query,
            "papers_found": len(final_state.papers),
            "synthesis": final_state.synthesis,
            "report": final_state.report,
            "iterations": final_state.iteration_count,
            "papers": final_state.papers,
            "vector_search_results": final_state.vector_db_results
        }

async def main():
    """Main function to run the research system"""
    orchestrator = ResearchOrchestrator()
    
    # Example research query
    research_query = "machine learning applications in drug discovery"
    
    print(f"Starting research on: {research_query}")
    print("=" * 50)
    
    results = await orchestrator.run_research(research_query)
    
    print(f"\nResearch completed!")
    print(f"Papers analyzed: {results['papers_found']}")
    print(f"Iterations: {results['iterations']}")
    print("\n" + "=" * 50)
    print("FINAL REPORT:")
    print("=" * 50)
    print(results['report'])
    
    # Save report to file
    with open(f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", "w") as f:
        f.write(results['report'])
    
    print(f"\nReport saved to research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

if __name__ == "__main__":
    asyncio.run(main())