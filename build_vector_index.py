import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass




@dataclass
class SearchResult:
    doc_id: str
    score: float
    rank: int
    title: str
    text_preview: str
    tags: List[str]





class VectorSearchEngine:
    """
    Multimodel Search engine
    support 3 diffrent model 
    
    """



    def __init__(self, embeddings_dir: str = "data/embeddings"):
        
        self.embeddings_dir = Path(embeddings_dir)
        print("📂 Loading embeddings and metadata...")

        self.short_embeddings = np.load(self.embeddings_dir / "embeddings_256d.npy")
        self.full_embeddings = np.load(self.embeddings_dir / "embeddings_768d.npy")

        # Load Document ID
        with open(self.embeddings_dir / "document_ids.json", 'r') as f:
            self.doc_ids = json.load(f)


        # Load original documents for metadata

        print("📂 Loading document metadata...")

        self.documents = self._load_documents()

        self.id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}

        print(f"✅ Loaded {len(self.doc_ids):,} documents")
        print(f"   - Short embeddings: {self.short_embeddings.shape}")
        print(f"   - Full embeddings: {self.full_embeddings.shape}")


        # Load embedding model for queries
        print("🤖 Loading embedding model for queries...")
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True
        )
        print("✅ Model loaded\n")

    
    def _load_documents(self) -> Dict[str, Dict]:
        """Load original documents for metadata"""
        documents = {}
        with open("data/processed_documents.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                documents[doc['id']] = doc
        return documents


    def embed_query(self, query: str, target_dim: int = None) -> np.ndarray:
        """
        Embed a search query.
        
        Args:
            query: Search query text
            target_dim: Target dimension (256 or 768). If None, returns full.
            
        Returns:
            Query embedding vector
        """
        # Generate full embedding
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Truncate if needed
        if target_dim and target_dim < len(embedding):
            embedding = embedding[:target_dim].copy()
            # Re-normalize after truncation
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding

    
    def cosine_similarity_search(
        self, 
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform cosine similarity search.
        
        Args:
            query_embedding: Query vector (normalized)
            embeddings: Database embeddings (normalized)
            top_k: Number of results to return
            
        Returns:
            (indices, scores) of top-k results
        """
        # Cosine similarity = dot product (since embeddings are normalized)
        scores = np.dot(embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_indices]
        
        return top_indices, top_scores


    
    def search_mode1_full(
        self, 
        query: str, 
        top_k: int = 10
    ) -> Tuple[List[SearchResult], Dict]:
        """
        MODE 1: Full 768-dim search (baseline).
        Single-stage search using full embeddings.
        
        Returns:
            (results, metrics)
        """
        start_time = time.time()
        
        # Embed query with full dimensions
        query_emb = self.embed_query(query, target_dim=768)
        embed_time = time.time() - start_time
        
        # Search full embeddings
        search_start = time.time()
        indices, scores = self.cosine_similarity_search(
            query_emb,
            self.full_embeddings,
            top_k=top_k
        )
        search_time = time.time() - search_start
        
        # Format results
        results = self._format_results(indices, scores)
        
        total_time = time.time() - start_time
        
        metrics = {
            'mode': 'full_768d',
            'total_time_ms': total_time * 1000,
            'embed_time_ms': embed_time * 1000,
            'search_time_ms': search_time * 1000,
            'num_results': len(results),
            'embeddings_searched': len(self.full_embeddings)
        }
        
        return results, metrics

    
    def search_mode2_fast(
        self, 
        query: str, 
        top_k: int = 10
    ) -> Tuple[List[SearchResult], Dict]:
        """
        MODE 2: Fast 256-dim only search.
        Single-stage search using short embeddings.
        
        Returns:
            (results, metrics)
        """
        start_time = time.time()
        
        # Embed query with short dimensions
        query_emb = self.embed_query(query, target_dim=256)
        embed_time = time.time() - start_time
        
        # Search short embeddings
        search_start = time.time()
        indices, scores = self.cosine_similarity_search(
            query_emb,
            self.short_embeddings,
            top_k=top_k
        )
        search_time = time.time() - search_start
        
        # Format results
        results = self._format_results(indices, scores)
        
        total_time = time.time() - start_time
        
        metrics = {
            'mode': 'fast_256d',
            'total_time_ms': total_time * 1000,
            'embed_time_ms': embed_time * 1000,
            'search_time_ms': search_time * 1000,
            'num_results': len(results),
            'embeddings_searched': len(self.short_embeddings)
        }
        
        return results, metrics


    
    def search_mode3_mrl(
        self, 
        query: str, 
        top_k: int = 10,
        first_stage_k: int = 100
    ) -> Tuple[List[SearchResult], Dict]:
        """
        MODE 3: Two-stage MRL search (BEST PERFORMANCE).
        
        Stage 1: Fast 256-dim search → get top 100 candidates
        Stage 2: Rerank with 768-dim → get final top 10
        
        Args:
            query: Search query
            top_k: Final number of results (default 10)
            first_stage_k: Candidates from first stage (default 100)
        
        Returns:
            (results, metrics)
        """
        start_time = time.time()
        
        # Embed query (we need both dimensions)
        query_short = self.embed_query(query, target_dim=256)
        query_full = self.embed_query(query, target_dim=768)
        embed_time = time.time() - start_time
        
        # STAGE 1: Fast search with 256-dim
        stage1_start = time.time()
        candidate_indices, _ = self.cosine_similarity_search(
            query_short,
            self.short_embeddings,
            top_k=first_stage_k
        )
        stage1_time = time.time() - stage1_start
        
        # STAGE 2: Rerank candidates with 768-dim
        stage2_start = time.time()
        
        # Get full embeddings for candidates only
        candidate_full_embs = self.full_embeddings[candidate_indices]
        
        # Rerank with full embeddings
        rerank_scores = np.dot(candidate_full_embs, query_full)
        
        # Get top-k from reranked results
        top_rerank_indices = np.argsort(rerank_scores)[-top_k:][::-1]
        
        # Map back to original indices
        final_indices = candidate_indices[top_rerank_indices]
        final_scores = rerank_scores[top_rerank_indices]
        
        stage2_time = time.time() - stage2_start
        
        # Format results
        results = self._format_results(final_indices, final_scores)
        
        total_time = time.time() - start_time
        
        metrics = {
            'mode': 'mrl_two_stage',
            'total_time_ms': total_time * 1000,
            'embed_time_ms': embed_time * 1000,
            'stage1_time_ms': stage1_time * 1000,
            'stage2_time_ms': stage2_time * 1000,
            'num_results': len(results),
            'first_stage_candidates': first_stage_k,
            'embeddings_searched_stage1': len(self.short_embeddings),
            'embeddings_searched_stage2': first_stage_k
        }
        
        return results, metrics


    def _format_results(
        self, 
        indices: np.ndarray, 
        scores: np.ndarray
    ) -> List[SearchResult]:
        """Format search results with metadata"""
        results = []
        
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            
            # Create preview (first 150 chars)
            text_preview = doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text']
            
            result = SearchResult(
                doc_id=doc_id,
                score=float(score),
                rank=rank,
                title=doc['title'],
                text_preview=text_preview,
                tags=doc.get('tags', [])
            )
            
            results.append(result)
        
        return results

    
    def print_results(self, results: List[SearchResult], metrics: Dict):
        """Pretty print search results and metrics"""
        print(f"\n{'='*80}")
        print(f"🔍 Search Mode: {metrics['mode'].upper()}")
        print(f"{'='*80}")
        
        print(f"\n⏱️  Performance Metrics:")
        print(f"   Total time: {metrics['total_time_ms']:.2f} ms")
        if 'stage1_time_ms' in metrics:
            print(f"   Stage 1 (256-dim): {metrics['stage1_time_ms']:.2f} ms")
            print(f"   Stage 2 (768-dim rerank): {metrics['stage2_time_ms']:.2f} ms")
        else:
            print(f"   Search time: {metrics['search_time_ms']:.2f} ms")
        
        print(f"\n📊 Top {len(results)} Results:")
        print("-" * 80)
        
        for result in results:
            print(f"\n#{result.rank} | Score: {result.score:.4f}")
            print(f"📌 {result.title}")
            print(f"🏷️  Tags: {', '.join(result.tags[:5])}")
            print(f"📄 {result.text_preview}")
            print("-" * 80)



if __name__ == "__main__":
    
    # Initialize search engine
    print("🚀 Initializing Vector Search Engine...\n")
    search_engine = VectorSearchEngine()
    
    # Test queries
    # test_queries = [
        # "How to reverse a string in Python?",
        # "JavaScript async await tutorial",
        # "SQL join multiple tables",
    # ]

    test_queries =[" I always create a new empty database, after that backup and restore of the existing database into it, but is this really the best way? As it seems very error prone and over complicated for me "]
    
    print("\n" + "="*80)
    print("🧪 TESTING ALL 3 SEARCH MODES")
    print("="*80)
    
    for query in test_queries:
        print(f"\n\n{'#'*80}")
        print(f"Query: '{query}'")
        print(f"{'#'*80}")
        
        # MODE 1: Full search
        print("\n--- MODE 1: Full 768-dim Search ---")
        results1, metrics1 = search_engine.search_mode1_full(query, top_k=5)
        search_engine.print_results(results1, metrics1)
        
        # MODE 2: Fast search
        print("\n--- MODE 2: Fast 256-dim Search ---")
        results2, metrics2 = search_engine.search_mode2_fast(query, top_k=5)
        search_engine.print_results(results2, metrics2)
        
        # MODE 3: MRL two-stage
        print("\n--- MODE 3: Two-Stage MRL Search ---")
        results3, metrics3 = search_engine.search_mode3_mrl(query, top_k=5, first_stage_k=50)
        search_engine.print_results(results3, metrics3)
        
        # Performance comparison
        print("\n" + "="*80)
        print("📊 PERFORMANCE COMPARISON")
        print("="*80)
        print(f"Mode 1 (Full 768d):     {metrics1['total_time_ms']:.2f} ms")
        print(f"Mode 2 (Fast 256d):     {metrics2['total_time_ms']:.2f} ms")
        print(f"Mode 3 (MRL Two-Stage): {metrics3['total_time_ms']:.2f} ms")
        print(f"\nSpeedup (Mode 2 vs 1): {metrics1['total_time_ms'] / metrics2['total_time_ms']:.2f}x")
        print(f"Speedup (Mode 3 vs 1): {metrics1['total_time_ms'] / metrics3['total_time_ms']:.2f}x")
    
    print("\n\n" + "="*80)
    print("✅ PART 3 COMPLETE!")
    print("="*80)
    print("\n🎯 You now have a working vector search engine with 3 modes!")
    print("🚀 Next: Part 4 - Build comprehensive benchmarking system")