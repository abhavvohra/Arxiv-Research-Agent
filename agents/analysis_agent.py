from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import tiktoken
import re
import json
from datetime import datetime

class AnalysisAgent:
    """Enhanced agent for analyzing individual research papers with detailed insights"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = 8000  # Reserve tokens for response
    
    def analyze_paper(self, paper: Dict[str, Any], analysis_depth: str = "comprehensive") -> Dict[str, Any]:
        """Analyze a single paper with configurable depth"""
        
        try:
            # Determine analysis type
            if analysis_depth == "quick":
                analysis_result = self._quick_analysis(paper)
            elif analysis_depth == "comprehensive":
                analysis_result = self._comprehensive_analysis(paper)
            else:
                analysis_result = self._detailed_analysis(paper)
            
            return {
                "paper_info": paper,
                "analysis": analysis_result.get("analysis", ""),
                "structured_insights": analysis_result.get("insights", {}),
                "relevance_score": analysis_result.get("relevance_score", 5.0),
                "quality_score": analysis_result.get("quality_score", 5.0),
                "novelty_score": analysis_result.get("novelty_score", 5.0),
                "analyzed_at": datetime.now().isoformat(),
                "analysis_depth": analysis_depth
            }
            
        except Exception as e:
            return {
                "paper_info": paper,
                "analysis": f"Error analyzing paper: {str(e)}",
                "structured_insights": {},
                "relevance_score": 0.0,
                "quality_score": 0.0,
                "novelty_score": 0.0,
                "analyzed_at": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _comprehensive_analysis(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of a paper"""
        
        # Truncate abstract if too long
        abstract = self._truncate_text(paper.get("summary", ""), 1000)
        
        prompt = f"""
        Perform a comprehensive analysis of this research paper:
        
        Title: {paper.get('title', 'Unknown')}
        Authors: {', '.join(paper.get('authors', [])[:5])}
        Categories: {', '.join(paper.get('categories', []))}
        Published: {paper.get('published', 'Unknown')}
        Abstract: {abstract}
        
        Provide a detailed analysis including:
        
        1. MAIN CONTRIBUTION (What is the key innovation or finding?)
        2. METHODOLOGY (What approach/methods were used?)
        3. KEY FINDINGS (What are the main results?)
        4. STRENGTHS (What are the paper's strong points?)
        5. LIMITATIONS (What are the weaknesses or limitations?)
        6. SIGNIFICANCE (How important is this work?)
        7. PRACTICAL APPLICATIONS (What are the real-world applications?)
        8. RELATED WORK (How does it relate to existing research?)
        9. FUTURE WORK (What directions does it suggest?)
        10. TECHNICAL QUALITY (Assessment of technical rigor)
        
        Additionally, provide scores (1-10) for:
        - Relevance to current research trends
        - Technical quality and rigor
        - Novelty and innovation
        - Practical impact potential
        
        Format the response as JSON with this structure:
        {{
            "analysis": "comprehensive analysis text",
            "insights": {{
                "main_contribution": "...",
                "methodology": "...",
                "key_findings": "...",
                "strengths": "...",
                "limitations": "...",
                "significance": "...",
                "applications": "...",
                "related_work": "...",
                "future_work": "...",
                "technical_quality": "..."
            }},
            "relevance_score": X,
            "quality_score": X,
            "novelty_score": X,
            "impact_score": X
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._parse_analysis_response(response.content)
    
    def _detailed_analysis(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed analysis of a paper"""
        
        abstract = self._truncate_text(paper.get("summary", ""), 800)
        
        prompt = f"""
        Analyze this research paper in detail:
        
        Title: {paper.get('title', 'Unknown')}
        Authors: {', '.join(paper.get('authors', [])[:3])}
        Abstract: {abstract}
        
        Provide analysis covering:
        1. Main contribution and innovation
        2. Methodology and approach
        3. Key findings and results
        4. Strengths and weaknesses
        5. Significance and impact
        6. Practical applications
        
        Rate the paper (1-10) on:
        - Relevance to current research
        - Technical quality
        - Novelty and innovation
        
        Format as JSON with analysis text and scores.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._parse_analysis_response(response.content)
    
    def _quick_analysis(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Perform quick analysis of a paper"""
        
        abstract = self._truncate_text(paper.get("summary", ""), 500)
        
        prompt = f"""
        Provide a quick analysis of this paper:
        
        Title: {paper.get('title', 'Unknown')}
        Abstract: {abstract}
        
        In 3-4 sentences, summarize:
        1. What the paper is about
        2. Main contribution
        3. Key findings
        4. Significance
        
        Rate relevance (1-10) to current research trends.
        
        Format as JSON with brief analysis and relevance score.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._parse_analysis_response(response.content)
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the analysis response from LLM"""
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
            else:
                # Fallback: treat entire response as analysis text
                return {
                    "analysis": response_text,
                    "insights": {},
                    "relevance_score": 5.0,
                    "quality_score": 5.0,
                    "novelty_score": 5.0
                }
        except json.JSONDecodeError:
            return {
                "analysis": response_text,
                "insights": {},
                "relevance_score": 5.0,
                "quality_score": 5.0,
                "novelty_score": 5.0
            }
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to fit within token limits"""
        if len(text) <= max_length:
            return text
        
        # Try to find a good breaking point
        truncated = text[:max_length]
        last_sentence = truncated.rfind('.')
        if last_sentence > max_length * 0.7:  # If we can keep most of the text
            return truncated[:last_sentence + 1]
        
        return truncated + "..."
    
    def analyze_paper_batch(self, papers: List[Dict[str, Any]], 
                          analysis_depth: str = "detailed") -> List[Dict[str, Any]]:
        """Analyze multiple papers in batch"""
        
        results = []
        for paper in papers:
            try:
                analysis = self.analyze_paper(paper, analysis_depth)
                results.append(analysis)
            except Exception as e:
                results.append({
                    "paper_info": paper,
                    "analysis": f"Error in batch analysis: {str(e)}",
                    "error": str(e),
                    "analyzed_at": datetime.now().isoformat()
                })
        
        return results
    
    def extract_methodologies(self, analyzed_papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and categorize methodologies from analyzed papers"""
        
        methodologies = []
        
        for paper_analysis in analyzed_papers:
            insights = paper_analysis.get("structured_insights", {})
            methodology = insights.get("methodology", "")
            
            if methodology and methodology != "Not specified":
                methodologies.append({
                    "paper_title": paper_analysis.get("paper_info", {}).get("title", "Unknown"),
                    "methodology": methodology,
                    "quality_score": paper_analysis.get("quality_score", 0.0)
                })
        
        return methodologies
    
    def identify_research_trends(self, analyzed_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify research trends from analyzed papers"""
        
        # Extract key themes and contributions
        contributions = []
        applications = []
        limitations = []
        
        for paper_analysis in analyzed_papers:
            insights = paper_analysis.get("structured_insights", {})
            
            if insights.get("main_contribution"):
                contributions.append(insights["main_contribution"])
            if insights.get("applications"):
                applications.append(insights["applications"])
            if insights.get("limitations"):
                limitations.append(insights["limitations"])
        
        # Use LLM to identify trends
        prompt = f"""
        Based on these research contributions, identify key trends:
        
        Contributions:
        {chr(10).join([f"- {contrib}" for contrib in contributions[:10]])}
        
        Applications:
        {chr(10).join([f"- {app}" for app in applications[:10]])}
        
        Limitations:
        {chr(10).join([f"- {lim}" for lim in limitations[:10]])}
        
        Identify:
        1. Emerging themes (top 5)
        2. Popular methodologies (top 5)
        3. Common applications (top 5)
        4. Recurring limitations (top 5)
        5. Research directions (top 5)
        
        Format as JSON with these categories.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"trends": "Could not identify trends"}
        except:
            return {"trends": "Error identifying trends"}
    
    def compare_papers(self, paper1_analysis: Dict[str, Any], 
                      paper2_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two analyzed papers"""
        
        paper1_info = paper1_analysis.get("paper_info", {})
        paper2_info = paper2_analysis.get("paper_info", {})
        
        prompt = f"""
        Compare these two research papers:
        
        Paper 1: {paper1_info.get('title', 'Unknown')}
        Analysis: {paper1_analysis.get('analysis', '')[:500]}
        
        Paper 2: {paper2_info.get('title', 'Unknown')}
        Analysis: {paper2_analysis.get('analysis', '')[:500]}
        
        Compare them on:
        1. Methodology approaches
        2. Key findings
        3. Strengths and weaknesses
        4. Novelty and significance
        5. Practical applications
        
        Identify:
        - Similarities
        - Differences
        - Complementary aspects
        - Which is more impactful and why
        
        Format as structured comparison.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "paper1": paper1_info.get("title", "Unknown"),
            "paper2": paper2_info.get("title", "Unknown"),
            "comparison": response.content,
            "compared_at": datetime.now().isoformat()
        }
    
    def generate_analysis_summary(self, analyzed_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of all paper analyses"""
        
        total_papers = len(analyzed_papers)
        avg_relevance = sum(p.get("relevance_score", 0) for p in analyzed_papers) / total_papers if total_papers > 0 else 0
        avg_quality = sum(p.get("quality_score", 0) for p in analyzed_papers) / total_papers if total_papers > 0 else 0
        avg_novelty = sum(p.get("novelty_score", 0) for p in analyzed_papers) / total_papers if total_papers > 0 else 0
        
        high_quality_papers = [p for p in analyzed_papers if p.get("quality_score", 0) >= 7]
        high_relevance_papers = [p for p in analyzed_papers if p.get("relevance_score", 0) >= 7]
        
        return {
            "total_papers_analyzed": total_papers,
            "average_scores": {
                "relevance": avg_relevance,
                "quality": avg_quality,
                "novelty": avg_novelty
            },
            "high_quality_papers": len(high_quality_papers),
            "high_relevance_papers": len(high_relevance_papers),
            "analysis_completed_at": datetime.now().isoformat()
        }