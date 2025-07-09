import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import hashlib
import json
import numpy as np
from datetime import datetime
import os

class VectorDatabaseAgent:
    """Enhanced vector database agent for semantic search and paper organization"""
    
    def __init__(self, collection_name: str = "research_papers", persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use default embedding function (sentence-transformers)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add papers to the vector database with enhanced metadata"""
        
        if not papers:
            return {"added": 0, "skipped": 0, "errors": []}
        
        documents = []
        metadatas = []
        ids = []
        added_count = 0
        skipped_count = 0
        errors = []
        
        for i, paper in enumerate(papers):
            try:
                # Create unique ID based on paper content
                paper_id = self._generate_paper_id(paper)
                
                # Check if paper already exists
                if self._paper_exists(paper_id):
                    skipped_count += 1
                    continue
                
                # Create document text for embedding
                doc_text = self._create_document_text(paper)
                
                # Create enhanced metadata
                metadata = self._create_metadata(paper)
                
                documents.append(doc_text)
                metadatas.append(metadata)
                ids.append(paper_id)
                added_count += 1
                
            except Exception as e:
                errors.append(f"Error processing paper {i}: {str(e)}")
                continue
        
        # Add to collection if we have documents
        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                errors.append(f"Error adding to collection: {str(e)}")
                return {"added": 0, "skipped": skipped_count, "errors": errors}
        
        return {
            "added": added_count,
            "skipped": skipped_count,
            "errors": errors,
            "total_in_collection": self.collection.count()
        }
    
    def semantic_search(self, query: str, n_results: int = 5, 
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform enhanced semantic search with filtering"""
        
        try:
            # Build where clause from filters
            where_clause = self._build_where_clause(filters) if filters else None
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            for i in range(len(results["documents"][0])):
                result = {
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance_score": 1 - results["distances"][0][i],  # Convert distance to relevance
                    "rank": i + 1
                }
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            return [{"error": f"Search error: {str(e)}"}]
    
    def find_similar_papers(self, paper: Dict[str, Any], n_results: int = 5) -> List[Dict[str, Any]]:
        """Find papers similar to a given paper"""
        
        # Create query text from paper
        query_text = self._create_document_text(paper)
        
        # Get paper ID to exclude from results
        paper_id = self._generate_paper_id(paper)
        
        # Search for similar papers
        results = self.semantic_search(query_text, n_results + 1)  # +1 to account for self-match
        
        # Filter out the paper itself
        similar_papers = [r for r in results if r.get("metadata", {}).get("paper_id") != paper_id]
        
        return similar_papers[:n_results]
    
    def get_papers_by_category(self, category: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Get papers from a specific category"""
        
        filters = {"primary_category": category}
        return self.semantic_search("", n_results, filters)
    
    def get_recent_papers(self, days: int = 30, n_results: int = 10) -> List[Dict[str, Any]]:
        """Get recently added papers"""
        
        from datetime import datetime, timedelta
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        try:
            results = self.collection.get(
                where={"published": {"$gte": cutoff_date}},
                include=["documents", "metadatas"],
                limit=n_results
            )
            
            papers = []
            for i in range(len(results["documents"])):
                papers.append({
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "distance": 0.0,  # No distance for direct retrieval
                    "relevance_score": 1.0
                })
            
            return papers
            
        except Exception as e:
            return [{"error": f"Recent papers error: {str(e)}"}]
    
    def cluster_papers(self, n_clusters: int = 5) -> Dict[str, Any]:
        """Perform clustering of papers in the database"""
        
        try:
            # Get all papers
            all_papers = self.collection.get(include=["documents", "metadatas", "embeddings"])
            
            if not all_papers["embeddings"]:
                return {"error": "No embeddings available for clustering"}
            
            # Perform simple clustering using embeddings
            from sklearn.cluster import KMeans
            
            embeddings = np.array(all_papers["embeddings"])
            kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Group papers by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                
                clusters[label].append({
                    "document": all_papers["documents"][i],
                    "metadata": all_papers["metadatas"][i]
                })
            
            # Generate cluster summaries
            cluster_summaries = {}
            for cluster_id, papers in clusters.items():
                titles = [p["metadata"].get("title", "") for p in papers]
                cluster_summaries[cluster_id] = {
                    "size": len(papers),
                    "sample_titles": titles[:3],
                    "papers": papers
                }
            
            return {
                "clusters": cluster_summaries,
                "total_papers": len(all_papers["documents"]),
                "n_clusters": len(clusters)
            }
            
        except ImportError:
            return {"error": "scikit-learn not installed for clustering"}
        except Exception as e:
            return {"error": f"Clustering error: {str(e)}"}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        
        try:
            count = self.collection.count()
            
            if count == 0:
                return {"total_papers": 0, "categories": {}, "recent_additions": 0}
            
            # Get sample of papers for stats
            sample = self.collection.get(
                include=["metadatas"],
                limit=min(1000, count)
            )
            
            # Analyze categories
            categories = {}
            recent_count = 0
            cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            for metadata in sample["metadatas"]:
                # Count categories
                category = metadata.get("primary_category", "unknown")
                categories[category] = categories.get(category, 0) + 1
                
                # Count recent additions
                if metadata.get("added_at", "").split("T")[0] >= cutoff_date:
                    recent_count += 1
            
            return {
                "total_papers": count,
                "categories": categories,
                "recent_additions": recent_count,
                "sample_size": len(sample["metadatas"])
            }
            
        except Exception as e:
            return {"error": f"Stats error: {str(e)}"}
    
    def _generate_paper_id(self, paper: Dict[str, Any]) -> str:
        """Generate unique ID for a paper"""
        # Use title and URL to create unique identifier
        unique_string = f"{paper.get('title', '')}{paper.get('url', '')}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def _paper_exists(self, paper_id: str) -> bool:
        """Check if paper already exists in collection"""
        try:
            result = self.collection.get(ids=[paper_id])
            return len(result["ids"]) > 0
        except:
            return False
    
    def _create_document_text(self, paper: Dict[str, Any]) -> str:
        """Create text representation of paper for embedding"""
        
        title = paper.get("title", "")
        authors = ", ".join(paper.get("authors", []))
        abstract = paper.get("summary", "")
        categories = ", ".join(paper.get("categories", []))
        
        # Create structured text
        doc_text = f"""
        Title: {title}
        
        Authors: {authors}
        
        Categories: {categories}
        
        Abstract: {abstract}
        """.strip()
        
        return doc_text
    
    def _create_metadata(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for paper"""
        
        metadata = {
            "paper_id": self._generate_paper_id(paper),
            "title": paper.get("title", ""),
            "authors": ", ".join(paper.get("authors", [])),
            "url": paper.get("url", ""),
            "published": paper.get("published", ""),
            "primary_category": paper.get("primary_category", ""),
            "categories": ", ".join(paper.get("categories", [])),
            "pdf_url": paper.get("pdf_url", ""),
            "added_at": datetime.now().isoformat(),
            "quality_score": paper.get("quality_score", 0.0)
        }
        
        # Add optional fields if available
        if paper.get("doi"):
            metadata["doi"] = paper["doi"]
        if paper.get("journal_ref"):
            metadata["journal_ref"] = paper["journal_ref"]
        if paper.get("comment"):
            metadata["comment"] = paper["comment"]
        
        return metadata
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters"""
        
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, str):
                where_clause[key] = {"$eq": value}
            elif isinstance(value, list):
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict):
                where_clause[key] = value
            else:
                where_clause[key] = {"$eq": str(value)}
        
        return where_clause
    
    def clear_collection(self) -> Dict[str, Any]:
        """Clear all papers from the collection"""
        try:
            # Get all IDs
            all_ids = self.collection.get()["ids"]
            
            if all_ids:
                # Delete all documents
                self.collection.delete(ids=all_ids)
            
            return {"cleared": len(all_ids), "remaining": self.collection.count()}
            
        except Exception as e:
            return {"error": f"Clear error: {str(e)}"}
    
    def backup_collection(self, backup_path: str = None) -> Dict[str, Any]:
        """Backup collection to JSON file"""
        
        if backup_path is None:
            backup_path = f"backup_{self.collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Get all data
            all_data = self.collection.get(include=["documents", "metadatas"])
            
            # Save to file
            with open(backup_path, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            return {
                "backup_path": backup_path,
                "papers_backed_up": len(all_data["documents"]),
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Backup error: {str(e)}", "success": False}