from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
import re
from collections import Counter
from datetime import datetime

class ThemeExtractor:
    """Agent for extracting themes, patterns, and insights from analyzed papers"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
    
    def extract_themes(self, analyzed_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract comprehensive themes from analyzed papers"""
        
        if not analyzed_papers:
            return {"themes": [], "gaps": [], "contradictions": []}
        
        # Prepare paper content for analysis
        paper_content = self._prepare_content_for_analysis(analyzed_papers)
        
        # Extract themes using LLM
        themes_result = self._extract_themes_with_llm(paper_content)
        
        # Extract gaps and contradictions
        gaps_result = self._extract_research_gaps(paper_content)
        contradictions_result = self._extract_contradictions(paper_content)
        
        # Combine results
        return {
            "themes": themes_result.get("themes", []),
            "methodological_themes": themes_result.get("methodological_themes", []),
            "application_themes": themes_result.get("application_themes", []),
            "gaps": gaps_result,
            "contradictions": contradictions_result,
            "extracted_at": datetime.now().isoformat()
        }
    
    def _prepare_content_for_analysis(self, analyzed_papers: List[Dict[str, Any]]) -> str:
        """Prepare paper content for theme extraction"""
        
        content_parts = []
        
        for i, paper_analysis in enumerate(analyzed_papers, 1):
            paper_info = paper_analysis.get("paper_info", {})
            insights = paper_analysis.get("structured_insights", {})
            analysis = paper_analysis.get("analysis", "")
            
            content = f"""
            Paper {i}: {paper_info.get('title', 'Unknown')}
            Categories: {', '.join(paper_info.get('categories', []))}
            
            Main Contribution: {insights.get('main_contribution', 'Not specified')}
            Methodology: {insights.get('methodology', 'Not specified')}
            Key Findings: {insights.get('key_findings', 'Not specified')}
            Applications: {insights.get('applications', 'Not specified')}
            Limitations: {insights.get('limitations', 'Not specified')}
            
            Analysis Summary: {analysis[:300]}...
            """
            
            content_parts.append(content.strip())
        
        return "\n\n" + "="*40 + "\n\n".join(content_parts)
    
    def _extract_themes_with_llm(self, content: str) -> Dict[str, List[str]]:
        """Extract themes using LLM analysis"""
        
        prompt = f"""
        Analyze the following research papers and extract key themes:
        
        {content}
        
        Extract themes in these categories:
        
        1. RESEARCH THEMES (main topics and focus areas)
        2. METHODOLOGICAL THEMES (common approaches and techniques)
        3. APPLICATION THEMES (practical uses and domains)
        4. THEORETICAL THEMES (conceptual frameworks and theories)
        5. TECHNOLOGICAL THEMES (tools, technologies, and platforms)
        
        For each category, identify 3-5 most prominent themes.
        
        Format as JSON:
        {{
            "themes": ["theme1", "theme2", ...],
            "methodological_themes": ["method1", "method2", ...],
            "application_themes": ["app1", "app2", ...],
            "theoretical_themes": ["theory1", "theory2", ...],
            "technological_themes": ["tech1", "tech2", ...]
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"themes": ["Could not extract themes"]}
        except json.JSONDecodeError:
            return {"themes": ["Error parsing themes"]}
    
    def _extract_research_gaps(self, content: str) -> List[str]:
        """Extract research gaps from paper content"""
        
        prompt = f"""
        Based on the following research papers, identify key research gaps:
        
        {content}
        
        Look for:
        1. Unexplored areas mentioned by authors
        2. Limitations that suggest future work
        3. Contradictory findings that need resolution
        4. Methodological improvements needed
        5. Scalability and practical challenges
        6. Missing theoretical frameworks
        7. Underrepresented domains or populations
        
        List 5-7 most significant research gaps.
        Format as a JSON array of strings.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return ["Could not identify research gaps"]
        except json.JSONDecodeError:
            return ["Error parsing research gaps"]
    
    def _extract_contradictions(self, content: str) -> List[str]:
        """Extract contradictions and debates from paper content"""
        
        prompt = f"""
        Identify contradictions, debates, and conflicting findings from these papers:
        
        {content}
        
        Look for:
        1. Conflicting experimental results
        2. Disagreements in theoretical interpretations
        3. Different conclusions from similar studies
        4. Methodological debates
        5. Controversial claims or findings
        6. Inconsistent performance across different settings
        
        List 3-5 most significant contradictions or debates.
        Format as a JSON array of strings.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return ["No significant contradictions identified"]
        except json.JSONDecodeError:
            return ["Error parsing contradictions"]
    
    def extract_keyword_themes(self, analyzed_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract themes using keyword frequency analysis"""
        
        # Collect all text content
        all_text = []
        for paper_analysis in analyzed_papers:
            paper_info = paper_analysis.get("paper_info", {})
            analysis = paper_analysis.get("analysis", "")
            
            # Combine title, abstract, and analysis
            text_content = f"{paper_info.get('title', '')} {paper_info.get('summary', '')} {analysis}"
            all_text.append(text_content.lower())
        
        combined_text = " ".join(all_text)
        
        # Extract keywords using simple frequency analysis
        # (In a production system, you'd use more sophisticated NLP)
        words = re.findall(r'\b[a-z]{4,}\b', combined_text)
        
        # Filter out common words
        stop_words = {
            'this', 'that', 'with', 'from', 'they', 'been', 'have', 'their', 'said', 'each', 'which',
            'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first',
            'also', 'after', 'back', 'well', 'way', 'only', 'new', 'old', 'see', 'him', 'two',
            'how', 'its', 'our', 'out', 'day', 'get', 'may', 'say', 'she', 'use', 'her', 'all',
            'any', 'can', 'had', 'his', 'has', 'but', 'not', 'who', 'oil', 'sit', 'now', 'find',
            'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part', 'over', 'such',
            'take', 'than', 'them', 'well', 'were', 'will', 'paper', 'approach', 'method', 'study',
            'research', 'work', 'using', 'show', 'results', 'analysis', 'based', 'proposed'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 4]
        
        # Count frequency
        word_freq = Counter(filtered_words)
        
        # Get top keywords
        top_keywords = [word for word, count in word_freq.most_common(20)]
        
        return {
            "keyword_themes": top_keywords,
            "word_frequencies": dict(word_freq.most_common(10)),
            "total_words_analyzed": len(words),
            "unique_words": len(set(words))
        }
    
    def categorize_papers_by_theme(self, analyzed_papers: List[Dict[str, Any]], 
                                  themes: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize papers by extracted themes"""
        
        categorized = {theme: [] for theme in themes}
        uncategorized = []
        
        for paper_analysis in analyzed_papers:
            paper_info = paper_analysis.get("paper_info", {})
            analysis = paper_analysis.get("analysis", "").lower()
            
            # Find which themes this paper relates to
            paper_themes = []
            for theme in themes:
                if theme.lower() in analysis or theme.lower() in paper_info.get("summary", "").lower():
                    paper_themes.append(theme)
            
            if paper_themes:
                for theme in paper_themes:
                    categorized[theme].append({
                        "title": paper_info.get("title", "Unknown"),
                        "authors": paper_info.get("authors", []),
                        "relevance_score": paper_analysis.get("relevance_score", 0),
                        "quality_score": paper_analysis.get("quality_score", 0)
                    })
            else:
                uncategorized.append({
                    "title": paper_info.get("title", "Unknown"),
                    "authors": paper_info.get("authors", [])
                })
        
        return {
            "categorized": categorized,
            "uncategorized": uncategorized,
            "categorization_stats": {
                "total_papers": len(analyzed_papers),
                "categorized_papers": len(analyzed_papers) - len(uncategorized),
                "uncategorized_papers": len(uncategorized)
            }
        }
    
    def generate_theme_summary(self, theme_data: Dict[str, Any]) -> str:
        """Generate a summary of extracted themes"""
        
        themes = theme_data.get("themes", [])
        gaps = theme_data.get("gaps", [])
        contradictions = theme_data.get("contradictions", [])
        
        prompt = f"""
        Generate a comprehensive summary of the research themes:
        
        Main Themes:
        {chr(10).join([f"- {theme}" for theme in themes[:5]])}
        
        Research Gaps:
        {chr(10).join([f"- {gap}" for gap in gaps[:3]])}
        
        Contradictions/Debates:
        {chr(10).join([f"- {contradiction}" for contradiction in contradictions[:3]])}
        
        Create a summary that:
        1. Describes the overall landscape of research
        2. Highlights the most important themes
        3. Discusses how themes relate to each other
        4. Identifies key challenges and opportunities
        5. Suggests implications for future research
        
        Write in a clear, analytical style (300-400 words).
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def extract_temporal_themes(self, analyzed_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract themes showing temporal evolution"""
        
        # Group papers by year
        yearly_papers = {}
        for paper_analysis in analyzed_papers:
            paper_info = paper_analysis.get("paper_info", {})
            year = paper_info.get("published", "Unknown").split("-")[0]
            
            if year not in yearly_papers:
                yearly_papers[year] = []
            yearly_papers[year].append(paper_analysis)
        
        # Extract themes for each year
        yearly_themes = {}
        for year, papers in yearly_papers.items():
            if len(papers) >= 2:  # Only analyze years with multiple papers
                year_content = self._prepare_content_for_analysis(papers)
                year_themes = self._extract_themes_with_llm(year_content)
                yearly_themes[year] = year_themes.get("themes", [])
        
        return {
            "yearly_themes": yearly_themes,
            "temporal_analysis": self._analyze_temporal_trends(yearly_themes),
            "years_analyzed": list(yearly_themes.keys())
        }
    
    def _analyze_temporal_trends(self, yearly_themes: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze temporal trends in themes"""
        
        # Simple trend analysis
        all_themes = []
        for themes in yearly_themes.values():
            all_themes.extend(themes)
        
        theme_counts = Counter(all_themes)
        
        # Find emerging and declining themes
        years = sorted(yearly_themes.keys())
        if len(years) >= 3:
            recent_themes = []
            early_themes = []
            
            # Themes from recent years
            for year in years[-2:]:
                recent_themes.extend(yearly_themes.get(year, []))
            
            # Themes from early years
            for year in years[:2]:
                early_themes.extend(yearly_themes.get(year, []))
            
            emerging_themes = [theme for theme in set(recent_themes) if theme not in early_themes]
            declining_themes = [theme for theme in set(early_themes) if theme not in recent_themes]
        else:
            emerging_themes = []
            declining_themes = []
        
        return {
            "most_common_themes": [theme for theme, count in theme_counts.most_common(5)],
            "emerging_themes": emerging_themes[:3],
            "declining_themes": declining_themes[:3],
            "total_unique_themes": len(set(all_themes)),
            "years_span": len(years)
        }