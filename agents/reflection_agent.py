from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import re
import json

class ReflectionAgent:
    """Advanced reflection agent for iterative improvement of research synthesis"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.quality_criteria = {
            "completeness": [
                "comprehensive coverage of key aspects",
                "addresses all parts of the research query",
                "includes diverse perspectives",
                "covers methodology and findings"
            ],
            "accuracy": [
                "factual correctness",
                "proper interpretation of research",
                "correct citations and references",
                "logical consistency"
            ],
            "clarity": [
                "clear organization and structure",
                "easy to understand language",
                "logical flow of ideas",
                "effective use of examples"
            ],
            "depth": [
                "sufficient analytical depth",
                "meaningful insights and connections",
                "critical evaluation of findings",
                "identification of implications"
            ],
            "actionability": [
                "practical insights for researchers",
                "clear future directions",
                "specific recommendations",
                "identification of research gaps"
            ]
        }
    
    def reflect_on_synthesis(self, synthesis: str, query: str, iteration: int = 0) -> Dict[str, Any]:
        """Comprehensive reflection on synthesis quality with scoring and suggestions"""
        
        # Evaluate against quality criteria
        evaluation = self._evaluate_synthesis(synthesis, query)
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(synthesis, query, evaluation)
        
        # Calculate confidence and completeness scores
        confidence_score = self._calculate_confidence_score(evaluation)
        completeness_score = self._calculate_completeness_score(evaluation)
        
        # Generate detailed feedback
        feedback = self._generate_detailed_feedback(evaluation, suggestions, iteration)
        
        return {
            "feedback": feedback,
            "suggestions": suggestions,
            "confidence_score": confidence_score,
            "completeness_score": completeness_score,
            "evaluation": evaluation,
            "iteration": iteration
        }
    
    def _evaluate_synthesis(self, synthesis: str, query: str) -> Dict[str, Any]:
        """Evaluate synthesis against quality criteria"""
        
        prompt = f"""
        Evaluate the following research synthesis based on these quality criteria:
        
        Query: {query}
        Synthesis: {synthesis}
        
        Rate each criterion on a scale of 1-10 and provide specific observations:
        
        1. COMPLETENESS (1-10): Does it comprehensively cover all aspects?
        2. ACCURACY (1-10): Is the information factually correct and properly interpreted?
        3. CLARITY (1-10): Is it well-organized and easy to understand?
        4. DEPTH (1-10): Does it provide meaningful insights and analysis?
        5. ACTIONABILITY (1-10): Does it offer practical insights for researchers?
        
        For each criterion, provide:
        - Score (1-10)
        - Specific observations
        - Areas for improvement
        
        Format your response as JSON with this structure:
        {{
            "completeness": {{"score": X, "observations": "...", "improvements": "..."}},
            "accuracy": {{"score": X, "observations": "...", "improvements": "..."}},
            "clarity": {{"score": X, "observations": "...", "improvements": "..."}},
            "depth": {{"score": X, "observations": "...", "improvements": "..."}},
            "actionability": {{"score": X, "observations": "...", "improvements": "..."}}
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._create_default_evaluation()
        except json.JSONDecodeError:
            return self._create_default_evaluation()
    
    def _generate_improvement_suggestions(self, synthesis: str, query: str, evaluation: Dict[str, Any]) -> List[str]:
        """Generate specific improvement suggestions based on evaluation"""
        
        # Identify low-scoring areas
        low_scoring_areas = []
        for criterion, data in evaluation.items():
            if data.get("score", 0) < 7:
                low_scoring_areas.append(criterion)
        
        prompt = f"""
        Based on the evaluation of this research synthesis, provide specific, actionable improvement suggestions:
        
        Query: {query}
        Synthesis: {synthesis}
        
        Low-scoring areas: {', '.join(low_scoring_areas)}
        
        Provide 3-5 specific suggestions for improvement. Each suggestion should be:
        1. Specific and actionable
        2. Focused on the identified weaknesses
        3. Practical to implement
        
        Format as a JSON array of strings:
        ["suggestion 1", "suggestion 2", "suggestion 3"]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return ["Improve overall structure and organization"]
        except json.JSONDecodeError:
            return ["Improve overall structure and organization"]
    
    def _calculate_confidence_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate confidence score based on evaluation"""
        scores = [data.get("score", 0) for data in evaluation.values()]
        if not scores:
            return 0.0
        
        # Weighted average with emphasis on accuracy and completeness
        weights = {
            "accuracy": 0.3,
            "completeness": 0.25,
            "depth": 0.2,
            "clarity": 0.15,
            "actionability": 0.1
        }
        
        weighted_score = 0
        for criterion, weight in weights.items():
            score = evaluation.get(criterion, {}).get("score", 0)
            weighted_score += score * weight
        
        return min(weighted_score / 10.0, 1.0)
    
    def _calculate_completeness_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate completeness score"""
        completeness_score = evaluation.get("completeness", {}).get("score", 0)
        depth_score = evaluation.get("depth", {}).get("score", 0)
        
        # Completeness is average of completeness and depth scores
        return ((completeness_score + depth_score) / 2) / 10.0
    
    def _generate_detailed_feedback(self, evaluation: Dict[str, Any], suggestions: List[str], iteration: int) -> str:
        """Generate detailed feedback summary"""
        
        feedback_parts = []
        
        # Overall assessment
        avg_score = sum(data.get("score", 0) for data in evaluation.values()) / len(evaluation)
        feedback_parts.append(f"Overall Quality Score: {avg_score:.1f}/10")
        
        # Iteration context
        if iteration > 0:
            feedback_parts.append(f"Iteration {iteration} - Continuing improvement process")
        
        # Strengths
        strengths = []
        for criterion, data in evaluation.items():
            if data.get("score", 0) >= 8:
                strengths.append(f"{criterion.capitalize()}: {data.get('observations', '')}")
        
        if strengths:
            feedback_parts.append("Strengths:")
            feedback_parts.extend([f"• {strength}" for strength in strengths])
        
        # Areas for improvement
        weaknesses = []
        for criterion, data in evaluation.items():
            if data.get("score", 0) < 7:
                weaknesses.append(f"{criterion.capitalize()}: {data.get('improvements', '')}")
        
        if weaknesses:
            feedback_parts.append("Areas for Improvement:")
            feedback_parts.extend([f"• {weakness}" for weakness in weaknesses])
        
        # Improvement suggestions
        if suggestions:
            feedback_parts.append("Specific Suggestions:")
            feedback_parts.extend([f"• {suggestion}" for suggestion in suggestions])
        
        return "\n".join(feedback_parts)
    
    def _create_default_evaluation(self) -> Dict[str, Any]:
        """Create default evaluation when parsing fails"""
        return {
            "completeness": {"score": 5, "observations": "Evaluation needed", "improvements": "Add more comprehensive coverage"},
            "accuracy": {"score": 7, "observations": "Generally accurate", "improvements": "Verify all claims"},
            "clarity": {"score": 6, "observations": "Could be clearer", "improvements": "Improve organization"},
            "depth": {"score": 5, "observations": "Surface-level analysis", "improvements": "Add deeper insights"},
            "actionability": {"score": 5, "observations": "Limited actionable insights", "improvements": "Add practical recommendations"}
        }
    
    def reflect_on_paper_analysis(self, analysis: Dict[str, Any], paper: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on individual paper analysis quality"""
        
        prompt = f"""
        Evaluate the quality of this paper analysis:
        
        Paper Title: {paper.get('title', 'Unknown')}
        Analysis: {analysis.get('analysis', '')}
        
        Rate the analysis on:
        1. Comprehensiveness (1-10)
        2. Accuracy (1-10)
        3. Insight Quality (1-10)
        4. Relevance Assessment (1-10)
        
        Provide brief feedback and suggestions for improvement.
        
        Format as JSON:
        {{
            "scores": {{"comprehensiveness": X, "accuracy": X, "insight_quality": X, "relevance": X}},
            "feedback": "...",
            "suggestions": ["suggestion1", "suggestion2"]
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "scores": {"comprehensiveness": 5, "accuracy": 7, "insight_quality": 5, "relevance": 6},
                    "feedback": "Analysis needs improvement",
                    "suggestions": ["Add more depth", "Improve clarity"]
                }
        except json.JSONDecodeError:
            return {
                "scores": {"comprehensiveness": 5, "accuracy": 7, "insight_quality": 5, "relevance": 6},
                "feedback": "Analysis needs improvement",
                "suggestions": ["Add more depth", "Improve clarity"]
            }
    
    def generate_iteration_summary(self, iteration_history: List[Dict[str, Any]]) -> str:
        """Generate summary of iteration improvements"""
        
        if not iteration_history:
            return "No iteration history available"
        
        summary_parts = []
        summary_parts.append("Iteration Improvement Summary:")
        summary_parts.append("=" * 35)
        
        for i, iteration in enumerate(iteration_history, 1):
            confidence = iteration.get("confidence_score", 0)
            completeness = iteration.get("completeness_score", 0)
            
            summary_parts.append(f"Iteration {i}:")
            summary_parts.append(f"  Confidence: {confidence:.2f}")
            summary_parts.append(f"  Completeness: {completeness:.2f}")
            
            if iteration.get("suggestions"):
                summary_parts.append("  Key Improvements:")
                for suggestion in iteration["suggestions"][:2]:  # Show top 2
                    summary_parts.append(f"    • {suggestion}")
            
            summary_parts.append("")
        
        # Show overall improvement
        if len(iteration_history) > 1:
            first_confidence = iteration_history[0].get("confidence_score", 0)
            last_confidence = iteration_history[-1].get("confidence_score", 0)
            improvement = last_confidence - first_confidence
            
            summary_parts.append(f"Overall Improvement: {improvement:+.2f}")
        
        return "\n".join(summary_parts)