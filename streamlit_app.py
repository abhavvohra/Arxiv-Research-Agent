import streamlit as st
import asyncio
import json
from datetime import datetime
from main import ResearchOrchestrator

st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("🔬 Automated Academic Research System")
st.markdown("*Powered by LangGraph Multi-Agent Architecture*")

# Initialize session state
if 'research_results' not in st.session_state:
    st.session_state.research_results = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    max_papers = st.slider("Max Papers to Retrieve", 5, 20, 10)
    max_iterations = st.slider("Max Reflection Iterations", 1, 5, 3)
    
    st.header("📊 Research Statistics")
    if st.session_state.research_results:
        results = st.session_state.research_results
        st.metric("Papers Found", results.get('papers_found', 0))
        st.metric("Iterations", results.get('iterations', 0))
    
    st.header("📚 Research History")
    for i, query in enumerate(st.session_state.research_history[-5:]):
        st.text(f"{i+1}. {query[:50]}...")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Research Query")
    research_query = st.text_area(
        "Enter your research question:",
        placeholder="e.g., machine learning applications in drug discovery",
        height=100
    )
    
    if st.button("🚀 Start Research", type="primary"):
        if research_query:
            with st.spinner("Conducting research... This may take a few minutes."):
                try:
                    orchestrator = ResearchOrchestrator()
                    results = asyncio.run(orchestrator.run_research(
                        research_query, max_iterations=max_iterations
                    ))
                    st.session_state.research_results = results
                    st.session_state.research_history.append(research_query)
                    st.success("Research completed successfully!")
                except Exception as e:
                    st.error(f"Error during research: {str(e)}")
        else:
            st.error("Please enter a research query")

with col2:
    st.header("💡 Example Queries")
    example_queries = [
        "transformer models in natural language processing",
        "quantum computing applications in cryptography",
        "deep learning for medical image analysis",
        "reinforcement learning in robotics",
        "federated learning privacy techniques"
    ]
    
    for query in example_queries:
        if st.button(f"📄 {query}", key=f"example_{query}"):
            st.rerun()

# Display results
if st.session_state.research_results:
    results = st.session_state.research_results
    
    st.header("📋 Research Results")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📑 Full Report", "📚 Papers", "🔍 Vector Search"])
    
    with tab1:
        st.subheader("Research Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Query", results['query'])
        with col2:
            st.metric("Papers Analyzed", results['papers_found'])
        with col3:
            st.metric("Reflection Iterations", results['iterations'])
        
        st.subheader("Key Synthesis")
        st.write(results['synthesis'])
    
    with tab2:
        st.subheader("Complete Research Report")
        st.markdown(results['report'])
        
        # Download button
        st.download_button(
            label="📥 Download Report",
            data=results['report'],
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    with tab3:
        st.subheader("Analyzed Papers")
        for i, paper in enumerate(results['papers'], 1):
            with st.expander(f"Paper {i}: {paper['title']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Authors:** {', '.join(paper['authors'])}")
                    st.write(f"**Published:** {paper['published']}")
                    st.write(f"**Categories:** {', '.join(paper['categories'])}")
                    st.write(f"**Abstract:** {paper['summary']}")
                with col2:
                    st.link_button("🔗 View Paper", paper['url'])
                    st.link_button("📄 PDF", paper['pdf_url'])
    
    with tab4:
        st.subheader("Vector Database Search Results")
        if results.get('vector_search_results'):
            for i, result in enumerate(results['vector_search_results'], 1):
                with st.expander(f"Result {i} (Distance: {result['distance']:.3f})"):
                    st.write(result['document'])
                    st.json(result['metadata'])
        else:
            st.info("No vector search results available")

# Footer
st.markdown("---")
st.markdown("Built with LangGraph 🦜🔗 | Powered by OpenAI GPT-4 🤖 | ArXiv API 📚")