from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from datetime import datetime
import re

class ReportGenerator:
    """Advanced report generator for comprehensive research reports with proper citations"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.citation_style = "academic"
    
    def generate_comprehensive_report(self, 
                                    query: str,
                                    synthesis: str,
                                    papers: List[Dict[str, Any]],
                                    themes: List[str] = None,
                                    gaps: List[str] = None,
                                    contradictions: List[str] = None,
                                    clusters: List[Dict[str, Any]] = None,
                                    metadata: Dict[str, Any] = None) -> str:
        """Generate a comprehensive research report with all sections"""
        
        # Generate citations
        citations = self._generate_citations(papers)
        
        # Create report sections
        sections = {
            "executive_summary": self._generate_executive_summary(query, synthesis, metadata),
            "introduction": self._generate_introduction(query, len(papers)),
            "literature_review": self._generate_literature_review(papers, themes, clusters),
            "key_findings": self._generate_key_findings(synthesis, themes),
            "analysis_discussion": self._generate_analysis_discussion(synthesis, contradictions),
            "research_gaps": self._generate_research_gaps(gaps),
            "future_directions": self._generate_future_directions(synthesis, gaps),
            "methodology": self._generate_methodology_section(metadata),
            "conclusions": self._generate_conclusions(query, synthesis),
            "references": self._format_references(citations)
        }
        
        # Compile full report
        return self._compile_full_report(sections, metadata)
    
    def _generate_executive_summary(self, query: str, synthesis: str, metadata: Dict[str, Any]) -> str:
        """Generate executive summary section"""
        
        papers_count = metadata.get("papers_analyzed", 0) if metadata else 0
        confidence_score = metadata.get("confidence_score", 0.0) if metadata else 0.0
        
        prompt = f"""
        Create an executive summary for a research report on "{query}".
        
        Research Synthesis:
        {synthesis}
        
        Include:
        - Research objective and scope
        - Key findings (3-4 main points)
        - Methodology overview ({papers_count} papers analyzed)
        - Confidence level ({confidence_score:.1%})
        - Main implications
        
        Keep it concise (300-400 words) and suitable for academic or professional audiences.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_introduction(self, query: str, paper_count: int) -> str:
        """Generate introduction section"""
        
        prompt = f"""
        Write an introduction for a research report on "{query}".
        
        The introduction should:
        - Establish the research context and importance
        - Define the research question/objective
        - Outline the scope (analysis of {paper_count} papers)
        - Preview the report structure
        - Highlight the significance of the research area
        
        Write in academic style, approximately 400-500 words.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_literature_review(self, papers: List[Dict[str, Any]], 
                                  themes: List[str], 
                                  clusters: List[Dict[str, Any]]) -> str:
        """Generate literature review section"""
        
        # Organize papers by categories or themes
        paper_summaries = []
        for i, paper in enumerate(papers[:10], 1):  # Limit to top 10 papers
            authors = ", ".join(paper.get("authors", [])[:3])
            if len(paper.get("authors", [])) > 3:
                authors += " et al."
            
            summary = f"[{i}] {authors} ({paper.get('published', 'Unknown')}) - {paper.get('title', 'Unknown title')}"
            paper_summaries.append(summary)
        
        themes_text = ", ".join(themes[:5]) if themes else "Various themes identified"
        
        prompt = f"""
        Generate a literature review section based on the following papers:
        
        {chr(10).join(paper_summaries)}
        
        Key themes identified: {themes_text}
        
        Structure the review to:
        1. Overview of the literature landscape
        2. Thematic organization of findings
        3. Methodological approaches used
        4. Evolution of research over time
        5. Theoretical frameworks employed
        
        Use proper academic citations [1], [2], etc. and maintain scholarly tone.
        Approximately 800-1000 words.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_key_findings(self, synthesis: str, themes: List[str]) -> str:
        """Generate key findings section"""
        
        themes_text = "\n".join([f"- {theme}" for theme in themes[:5]]) if themes else "No specific themes provided"
        
        prompt = f"""
        Extract and organize the key findings from this research synthesis:
        
        {synthesis}
        
        Key themes to address:
        {themes_text}
        
        Organize findings into:
        1. Major discoveries and insights
        2. Consistent patterns across studies
        3. Quantitative results and trends
        4. Qualitative observations
        5. Methodological findings
        
        Present findings clearly with supporting evidence and citations where appropriate.
        Approximately 600-800 words.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_analysis_discussion(self, synthesis: str, contradictions: List[str]) -> str:
        """Generate analysis and discussion section"""
        
        contradictions_text = "\n".join([f"- {contradiction}" for contradiction in contradictions[:5]]) if contradictions else "No major contradictions identified"
        
        prompt = f"""
        Create an analysis and discussion section based on:
        
        Research Synthesis:
        {synthesis}
        
        Contradictions/Debates:
        {contradictions_text}
        
        The discussion should:
        1. Interpret the findings in context
        2. Address contradictions and debates
        3. Explain implications of results
        4. Connect findings to broader theory
        5. Discuss limitations and considerations
        6. Compare with existing knowledge
        
        Maintain analytical depth and scholarly rigor.
        Approximately 800-1000 words.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_research_gaps(self, gaps: List[str]) -> str:
        """Generate research gaps section"""
        
        if not gaps:
            gaps = ["Further analysis needed to identify specific research gaps"]
        
        gaps_text = "\n".join([f"- {gap}" for gap in gaps[:7]])
        
        prompt = f"""
        Expand on these research gaps to create a comprehensive section:
        
        {gaps_text}
        
        For each gap:
        1. Describe the gap clearly
        2. Explain why it's important
        3. Suggest potential research approaches
        4. Identify required resources/methods
        5. Discuss expected contributions
        
        Approximately 500-600 words.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_future_directions(self, synthesis: str, gaps: List[str]) -> str:
        """Generate future research directions section"""
        
        prompt = f"""
        Based on this research synthesis and identified gaps, suggest future research directions:
        
        Synthesis:
        {synthesis}
        
        Research Gaps:
        {chr(10).join([f"- {gap}" for gap in gaps[:5]]) if gaps else "No specific gaps identified"}
        
        Provide:
        1. Immediate research opportunities
        2. Long-term research programs
        3. Methodological innovations needed
        4. Interdisciplinary collaborations
        5. Practical applications to develop
        6. Technology/tools requirements
        
        Make recommendations specific and actionable.
        Approximately 500-700 words.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_methodology_section(self, metadata: Dict[str, Any]) -> str:
        """Generate methodology section"""
        
        papers_count = metadata.get("papers_analyzed", 0) if metadata else 0
        iterations = metadata.get("iterations", 0) if metadata else 0
        confidence = metadata.get("confidence_score", 0.0) if metadata else 0.0
        
        prompt = f"""
        Create a methodology section describing the research approach used:
        
        Key parameters:
        - Papers analyzed: {papers_count}
        - Reflection iterations: {iterations}
        - Confidence score: {confidence:.1%}
        
        Describe:
        1. Data collection approach (ArXiv API search)
        2. Paper selection criteria
        3. Analysis methodology (multi-agent LangGraph system)
        4. Quality assurance (reflection agents)
        5. Synthesis approach
        6. Limitations and considerations
        
        Keep it clear and replicable.
        Approximately 400-500 words.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_conclusions(self, query: str, synthesis: str) -> str:
        """Generate conclusions section"""
        
        prompt = f"""
        Write a conclusions section for the research on "{query}".
        
        Based on synthesis:
        {synthesis}
        
        Include:
        1. Summary of main findings
        2. Answers to research question
        3. Broader implications
        4. Contribution to knowledge
        5. Practical applications
        6. Final recommendations
        
        Approximately 400-500 words.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_citations(self, papers: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate properly formatted citations"""
        
        citations = []
        for i, paper in enumerate(papers, 1):
            authors = paper.get("authors", [])
            author_str = self._format_authors(authors)
            
            citation = {
                "number": i,
                "authors": author_str,
                "title": paper.get("title", "Unknown title"),
                "journal": paper.get("journal_ref", "arXiv preprint"),
                "year": paper.get("published", "Unknown").split("-")[0],
                "url": paper.get("url", ""),
                "doi": paper.get("doi", ""),
                "arxiv_id": self._extract_arxiv_id(paper.get("url", ""))
            }
            citations.append(citation)
        
        return citations
    
    def _format_authors(self, authors: List[str]) -> str:
        """Format author list for citations"""
        if not authors:
            return "Unknown authors"
        
        if len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} and {authors[1]}"
        elif len(authors) <= 5:
            return ", ".join(authors[:-1]) + f", and {authors[-1]}"
        else:
            return f"{authors[0]} et al."
    
    def _extract_arxiv_id(self, url: str) -> str:
        """Extract ArXiv ID from URL"""
        match = re.search(r'arxiv\.org/abs/([^/]+)', url)
        return match.group(1) if match else ""
    
    def _format_references(self, citations: List[Dict[str, str]]) -> str:
        """Format references section"""
        
        references = []
        for citation in citations:
            if citation["journal"] == "arXiv preprint":
                ref = f"[{citation['number']}] {citation['authors']}. \"{citation['title']}\". arXiv preprint arXiv:{citation['arxiv_id']} ({citation['year']}). {citation['url']}"
            else:
                ref = f"[{citation['number']}] {citation['authors']}. \"{citation['title']}\". {citation['journal']} ({citation['year']}). {citation['url']}"
            
            if citation["doi"]:
                ref += f" DOI: {citation['doi']}"
            
            references.append(ref)
        
        return "\n".join(references)
    
    def _compile_full_report(self, sections: Dict[str, str], metadata: Dict[str, Any]) -> str:
        """Compile all sections into final report"""
        
        # Report header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        papers_count = metadata.get("papers_analyzed", 0) if metadata else 0
        
        report_parts = []
        
        # Title and metadata
        report_parts.append("# Automated Academic Research Report")
        report_parts.append(f"**Generated**: {timestamp}")
        report_parts.append(f"**Papers Analyzed**: {papers_count}")
        if metadata:
            report_parts.append(f"**Confidence Score**: {metadata.get('confidence_score', 0.0):.1%}")
            report_parts.append(f"**Iterations**: {metadata.get('iterations', 0)}")
        report_parts.append("")
        
        # Table of contents
        report_parts.append("## Table of Contents")
        report_parts.append("1. Executive Summary")
        report_parts.append("2. Introduction")
        report_parts.append("3. Methodology")
        report_parts.append("4. Literature Review")
        report_parts.append("5. Key Findings")
        report_parts.append("6. Analysis and Discussion")
        report_parts.append("7. Research Gaps")
        report_parts.append("8. Future Directions")
        report_parts.append("9. Conclusions")
        report_parts.append("10. References")
        report_parts.append("")
        
        # Add sections
        section_titles = {
            "executive_summary": "Executive Summary",
            "introduction": "Introduction",
            "methodology": "Methodology",
            "literature_review": "Literature Review",
            "key_findings": "Key Findings",
            "analysis_discussion": "Analysis and Discussion",
            "research_gaps": "Research Gaps",
            "future_directions": "Future Directions",
            "conclusions": "Conclusions",
            "references": "References"
        }
        
        for section_key, section_title in section_titles.items():
            report_parts.append(f"## {section_title}")
            report_parts.append(sections.get(section_key, "Content not available"))
            report_parts.append("")
        
        # Footer
        report_parts.append("---")
        report_parts.append("*This report was generated using an automated academic research system*")
        report_parts.append("*powered by LangGraph multi-agent architecture with ArXiv API integration.*")
        
        return "\n".join(report_parts)
    
    def generate_summary_report(self, query: str, synthesis: str, papers: List[Dict[str, Any]]) -> str:
        """Generate a shorter summary report"""
        
        citations = self._generate_citations(papers[:5])  # Top 5 papers only
        
        prompt = f"""
        Create a concise research summary report on "{query}".
        
        Research Synthesis:
        {synthesis}
        
        Papers analyzed: {len(papers)}
        
        Include:
        1. Brief overview (2-3 sentences)
        2. Key findings (3-4 bullet points)
        3. Main implications (2-3 sentences)
        4. Top recommendations (2-3 bullet points)
        
        Keep the entire report under 500 words.
        Include citations [1], [2], etc. for key points.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Add references
        references = self._format_references(citations)
        
        return f"{response.content}\n\n## References\n{references}"