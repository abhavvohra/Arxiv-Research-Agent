from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
import re
from datetime import datetime

class SynthesisAgent:
    """Advanced agent for synthesizing insights from multiple analyzed papers"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
    
    def synthesize_findings(self, analyzed_papers: List[Dict[str, Any]], 
                          query: str,
                          themes: List[str] = None,
                          gaps: List[str] = None) -> str:
        """Synthesize findings from multiple analyzed papers"""
        
        if not analyzed_papers:
            return "No papers available for synthesis."
        
        # Prepare paper summaries for synthesis
        paper_summaries = self._prepare_paper_summaries(analyzed_papers)
        
        # Create synthesis prompt
        prompt = self._create_synthesis_prompt(
            query, paper_summaries, themes, gaps
        )
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _prepare_paper_summaries(self, analyzed_papers: List[Dict[str, Any]]) -> str:
        """Prepare paper summaries for synthesis"""
        
        summaries = []
        
        for i, paper_analysis in enumerate(analyzed_papers, 1):
            paper_info = paper_analysis.get("paper_info", {})
            analysis = paper_analysis.get("analysis", "")
            insights = paper_analysis.get("structured_insights", {})
            
            summary = f"""
            Paper {i}: {paper_info.get('title', 'Unknown Title')}
            Authors: {', '.join(paper_info.get('authors', [])[:3])}
            Published: {paper_info.get('published', 'Unknown')}
            Relevance Score: {paper_analysis.get('relevance_score', 0)}/10
            Quality Score: {paper_analysis.get('quality_score', 0)}/10
            
            Key Insights:
            - Main Contribution: {insights.get('main_contribution', 'Not specified')}
            - Methodology: {insights.get('methodology', 'Not specified')}
            - Key Findings: {insights.get('key_findings', 'Not specified')}
            - Strengths: {insights.get('strengths', 'Not specified')}
            - Limitations: {insights.get('limitations', 'Not specified')}
            - Applications: {insights.get('applications', 'Not specified')}
            
            Analysis: {analysis[:500]}...
            """
            
            summaries.append(summary.strip())
        
        return "\n\n" + "="*50 + "\n\n".join(summaries)
    
    def _create_synthesis_prompt(self, query: str, paper_summaries: str, 
                               themes: List[str] = None, gaps: List[str] = None) -> str:
        """Create comprehensive synthesis prompt"""
        
        themes_section = ""
        if themes:
            themes_section = f"""
            Key Themes to Address:
            {chr(10).join([f"- {theme}" for theme in themes[:5]])}
            """
        
        gaps_section = ""
        if gaps:
            gaps_section = f"""
            Research Gaps to Consider:
            {chr(10).join([f"- {gap}" for gap in gaps[:5]])}
            """
        
        prompt = f"""
        Based on the analyzed research papers, provide a comprehensive synthesis for the query: "{query}"
        
        {themes_section}
        {gaps_section}
        
        Papers Analysis:
        {paper_summaries}
        
        Your synthesis should include:
        
        1. OVERVIEW AND CONTEXT
        - Current state of the field
        - Scope and focus of reviewed literature
        - Timeline and evolution of research
        
        2. COMMON THEMES AND PATTERNS
        - Recurring findings across papers
        - Methodological approaches used
        - Theoretical frameworks employed
        - Consistent results and conclusions
        
        3. CONTRADICTIONS AND DEBATES
        - Conflicting findings or interpretations
        - Methodological differences leading to different results
        - Ongoing debates in the field
        - Areas of uncertainty
        
        4. METHODOLOGICAL INSIGHTS
        - Popular and emerging methodologies
        - Strengths and limitations of different approaches
        - Best practices identified
        - Innovation in research methods
        
        5. KEY FINDINGS AND CONTRIBUTIONS
        - Most significant discoveries
        - Breakthrough results
        - Practical implications
        - Theoretical advances
        
        6. RESEARCH GAPS AND LIMITATIONS
        - Underexplored areas
        - Methodological limitations
        - Data availability issues
        - Scalability challenges
        
        7. FUTURE RESEARCH DIRECTIONS
        - Promising research avenues
        - Emerging technologies and methods
        - Interdisciplinary opportunities
        - Long-term research programs
        
        8. PRACTICAL IMPLICATIONS
        - Real-world applications
        - Industry relevance
        - Policy implications
        - Societal impact
        
        9. QUALITY ASSESSMENT
        - Overall quality of the literature
        - Reliability of findings
        - Strength of evidence
        - Methodological rigor
        
        10. SYNTHESIS CONCLUSIONS
        - Overall assessment of the field
        - Most important takeaways
        - Confidence level in findings
        - Recommendations for practitioners/researchers
        
        Write the synthesis in a clear, academic style that would be suitable for a research report.
        Connect insights across papers and provide a coherent narrative.
        Cite specific papers when making claims (e.g., "Paper 1 demonstrates...", "Papers 3 and 5 both found...").
        """
        
        return prompt
    
    def improve_synthesis(self, current_synthesis: str, 
                         reflection_feedback: str,
                         improvement_suggestions: List[str]) -> str:
        """Improve synthesis based on reflection feedback"""
        
        suggestions_text = "\n".join([f"- {suggestion}" for suggestion in improvement_suggestions])
        
        prompt = f"""
        Improve the following research synthesis based on the reflection feedback and suggestions:
        
        Current Synthesis:
        {current_synthesis}
        
        Reflection Feedback:
        {reflection_feedback}
        
        Improvement Suggestions:
        {suggestions_text}
        
        Please revise the synthesis to:
        1. Address all the feedback points
        2. Implement the suggested improvements
        3. Enhance clarity and organization
        4. Strengthen the arguments and evidence
        5. Improve completeness and depth
        
        Maintain the academic tone and structure while making the improvements.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def synthesize_by_theme(self, analyzed_papers: List[Dict[str, Any]], 
                          theme: str) -> str:
        """Synthesize findings for a specific theme"""
        
        # Filter papers relevant to the theme
        relevant_papers = []
        for paper_analysis in analyzed_papers:
            analysis_text = paper_analysis.get("analysis", "").lower()
            if theme.lower() in analysis_text:
                relevant_papers.append(paper_analysis)
        
        if not relevant_papers:
            return f"No papers found specifically addressing the theme: {theme}"
        
        paper_summaries = self._prepare_paper_summaries(relevant_papers)
        
        prompt = f"""
        Synthesize findings specifically related to the theme: "{theme}"
        
        Relevant Papers:
        {paper_summaries}
        
        Focus on:
        1. How different papers address this theme
        2. Consensus and disagreements regarding this theme
        3. Methodological approaches for studying this theme
        4. Key findings and insights related to this theme
        5. Gaps and future directions for this theme
        
        Provide a focused synthesis that thoroughly explores this specific theme.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def comparative_synthesis(self, analyzed_papers: List[Dict[str, Any]], 
                            comparison_aspects: List[str]) -> str:
        """Create a comparative synthesis focusing on specific aspects"""
        
        paper_summaries = self._prepare_paper_summaries(analyzed_papers)
        aspects_text = ", ".join(comparison_aspects)
        
        prompt = f"""
        Create a comparative synthesis focusing on these aspects: {aspects_text}
        
        Papers Analysis:
        {paper_summaries}
        
        For each aspect, compare and contrast how different papers address it:
        1. Identify different approaches and perspectives
        2. Highlight similarities and differences
        3. Evaluate the strength of evidence for each approach
        4. Identify the most promising directions
        
        Structure the synthesis with clear sections for each comparison aspect.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def generate_synthesis_metrics(self, synthesis: str, 
                                 analyzed_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate metrics about the synthesis quality"""
        
        # Calculate basic metrics
        word_count = len(synthesis.split())
        paper_count = len(analyzed_papers)
        
        # Calculate average scores
        avg_relevance = sum(p.get("relevance_score", 0) for p in analyzed_papers) / paper_count if paper_count > 0 else 0
        avg_quality = sum(p.get("quality_score", 0) for p in analyzed_papers) / paper_count if paper_count > 0 else 0
        
        # Count citations (simple heuristic)
        citation_count = len(re.findall(r'Paper \d+', synthesis))
        
        # Estimate comprehensiveness
        key_sections = [
            "themes", "patterns", "findings", "gaps", "limitations", 
            "future", "implications", "methodology", "conclusions"
        ]
        
        sections_covered = sum(1 for section in key_sections if section.lower() in synthesis.lower())
        comprehensiveness = sections_covered / len(key_sections)
        
        return {
            "word_count": word_count,
            "papers_synthesized": paper_count,
            "citation_count": citation_count,
            "average_paper_relevance": avg_relevance,
            "average_paper_quality": avg_quality,
            "comprehensiveness_score": comprehensiveness,
            "sections_covered": sections_covered,
            "total_sections": len(key_sections),
            "generated_at": datetime.now().isoformat()
        }
    
    def extract_key_insights(self, synthesis: str) -> Dict[str, List[str]]:
        """Extract key insights from synthesis"""
        
        prompt = f"""
        Extract key insights from this research synthesis:
        
        {synthesis}
        
        Extract and categorize insights into:
        1. Main findings (top 5)
        2. Methodological insights (top 3)
        3. Practical applications (top 3)
        4. Research gaps (top 3)
        5. Future directions (top 3)
        
        Format as JSON with these categories as keys and lists of insights as values.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"insights": ["Could not extract structured insights"]}
        except json.JSONDecodeError:
            return {"insights": ["Error parsing insights"]}
    
    def create_executive_summary(self, synthesis: str, query: str) -> str:
        """Create executive summary of the synthesis"""
        
        prompt = f"""
        Create an executive summary of this research synthesis for the query: "{query}"
        
        Synthesis:
        {synthesis}
        
        The executive summary should:
        1. Be 200-300 words
        2. Highlight the most important findings
        3. Mention key implications
        4. Be accessible to non-experts
        5. Include confidence level assessment
        
        Write in a clear, professional tone suitable for decision-makers.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def validate_synthesis_quality(self, synthesis: str, 
                                 analyzed_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of the synthesis"""
        
        prompt = f"""
        Evaluate the quality of this research synthesis:
        
        {synthesis}
        
        Based on {len(analyzed_papers)} analyzed papers.
        
        Rate (1-10) on:
        1. Completeness - covers all important aspects
        2. Accuracy - correctly represents the literature
        3. Coherence - logical flow and organization
        4. Depth - sufficient analytical depth
        5. Balance - fair representation of different perspectives
        6. Clarity - easy to understand and well-written
        
        Provide scores and brief explanations for each criterion.
        Format as JSON with scores and explanations.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"quality_assessment": "Could not assess quality"}
        except json.JSONDecodeError:
            return {"quality_assessment": "Error in quality assessment"}