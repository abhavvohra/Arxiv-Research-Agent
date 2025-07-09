# Automated Academic Research System

An automated academic research system using LangGraph's multi-agent architecture, integrating ArXiv API with StateGraph orchestration to streamline scholarly paper retrieval, analysis, and synthesis for researchers and academics.

## 🚀 Features

### Core Capabilities
- **Multi-Agent Architecture**: Built with LangGraph StateGraph for orchestrated workflow
- **ArXiv Integration**: Comprehensive paper retrieval and metadata extraction
- **Semantic Search**: Vector database integration with ChromaDB for intelligent paper discovery
- **Reflection Agents**: Iterative improvement through quality assessment and feedback
- **Structured Reports**: Professional research reports with proper citations
- **Theme Extraction**: Automatic identification of research themes and gaps

### Advanced Features
- **Quality Filtering**: Automatic paper quality assessment and filtering
- **Semantic Clustering**: Intelligent grouping of papers by themes
- **Contradiction Detection**: Identification of conflicting findings and debates
- **Temporal Analysis**: Evolution of research themes over time
- **Comprehensive Citations**: Academic-style references with DOI support
- **Interactive Web Interface**: Streamlit-based user interface

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Required packages (see `requirements.txt`)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Arxiv agent"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `LANGCHAIN_API_KEY`: Your LangChain API key (optional)
- `LANGCHAIN_TRACING_V2`: Enable LangChain tracing (optional)

## 🚀 Usage

### Command Line Interface

Run the main research system:
```bash
python main.py
```

### Web Interface

Launch the Streamlit application:
```bash
streamlit run streamlit_app.py
```

### Programmatic Usage

```python
from main import ResearchOrchestrator
import asyncio

async def main():
    orchestrator = ResearchOrchestrator()
    
    # Run research on a topic
    results = await orchestrator.run_research(
        query="machine learning applications in drug discovery",
        max_iterations=3
    )
    
    print(f"Found {results['papers_found']} papers")
    print(f"Generated report: {results['report']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🏗️ Architecture

### Multi-Agent System

The system consists of several specialized agents:

1. **ArxivAgent**: Paper retrieval and metadata extraction
2. **VectorDatabaseAgent**: Semantic search and similarity matching
3. **AnalysisAgent**: Individual paper analysis and insight extraction
4. **SynthesisAgent**: Multi-paper synthesis and theme integration
5. **ReflectionAgent**: Quality assessment and iterative improvement
6. **ReportGenerator**: Professional report generation with citations
7. **ThemeExtractor**: Theme identification and trend analysis

### StateGraph Workflow

The research process follows a structured workflow:

```
Initialize → Retrieve Papers → Quality Filter → Vector Search → 
Semantic Clustering → Analyze Papers → Extract Themes → 
Synthesize → Reflect → [Iterate] → Generate Report → Finalize
```

### Data Flow

1. **Input**: Research query and parameters
2. **Retrieval**: ArXiv API search and filtering
3. **Processing**: Vector embedding and semantic analysis
4. **Analysis**: Individual paper analysis and scoring
5. **Synthesis**: Cross-paper insights and theme extraction
6. **Reflection**: Quality assessment and improvement
7. **Output**: Comprehensive research report

## 📊 Output Examples

### Research Report Structure

1. **Executive Summary**
2. **Introduction**
3. **Methodology**
4. **Literature Review**
5. **Key Findings**
6. **Analysis and Discussion**
7. **Research Gaps**
8. **Future Directions**
9. **Conclusions**
10. **References**

### Metrics and Scores

- **Relevance Score**: 1-10 rating of paper relevance
- **Quality Score**: 1-10 assessment of paper quality
- **Confidence Score**: Overall confidence in findings
- **Completeness Score**: Coverage of research aspects

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Test coverage includes:
- Unit tests for all agents
- Integration tests for workflow
- Mock tests for external APIs
- Performance and reliability tests

## 🔧 Configuration

### System Parameters

- `max_papers`: Maximum papers to retrieve (default: 10)
- `max_iterations`: Maximum reflection iterations (default: 3)
- `search_categories`: ArXiv categories to search (default: ["cs.AI", "cs.LG"])
- `analysis_depth`: Analysis depth level ("quick", "detailed", "comprehensive")

### Model Configuration

- **Analysis Agent**: GPT-4o-mini for cost-effective analysis
- **Synthesis Agent**: GPT-4o for high-quality synthesis
- **Reflection Agent**: GPT-4o for thorough quality assessment
- **Report Generator**: GPT-4o for professional reports

## 📁 Project Structure

```
Arxiv agent/
├── main.py                 # Main orchestrator
├── streamlit_app.py        # Web interface
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── README.md              # Documentation
├── agents/                # Agent implementations
│   ├── arxiv_agent.py     # ArXiv API integration
│   ├── vector_agent.py    # Vector database management
│   ├── analysis_agent.py  # Paper analysis
│   ├── synthesis_agent.py # Multi-paper synthesis
│   ├── reflection_agent.py # Quality reflection
│   ├── report_generator.py # Report generation
│   ├── theme_extractor.py # Theme extraction
│   └── state_graph.py     # StateGraph orchestration
├── tests/                 # Test suite
│   └── test_agents.py     # Agent tests
└── chroma_db/            # Vector database storage
```

## 🔬 Research Capabilities

### Paper Analysis
- Comprehensive analysis of individual papers
- Methodology assessment and categorization
- Strength and limitation identification
- Novelty and significance evaluation

### Theme Extraction
- Automatic identification of research themes
- Methodological pattern recognition
- Application domain categorization
- Temporal trend analysis

### Synthesis Generation
- Cross-paper insight synthesis
- Contradiction and debate identification
- Research gap analysis
- Future direction recommendations

### Quality Assurance
- Iterative improvement through reflection
- Confidence and completeness scoring
- Bias detection and mitigation
- Citation accuracy verification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangGraph**: Multi-agent orchestration framework
- **ArXiv**: Open access to scientific papers
- **ChromaDB**: Vector database for semantic search
- **OpenAI**: Language models for analysis and synthesis
- **Streamlit**: Web interface framework

## 📞 Support

For issues and questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with details
4. Contact the maintainers

---

*Generated with LangGraph Multi-Agent Architecture 🤖*