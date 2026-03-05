import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from typing import List, Dict
import time


class EmbeddingGenerator :
    """
    Generate embeddings using open-source Hugging Face models.
    Supports Matryoshka Representation Learning (MRL) for multiple dimensions.
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Hugging Face model identifier
        """
        print(f"Loading model: {model_name}")
        
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model with trust_remote_code for custom models
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=self.device
        )
        
        # Get model's native dimension
        self.full_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded - Full dimension: {self.full_dim}")
        
        # For nomic-embed: supports 64, 128, 256, 512, 768
        # We'll use 256 (fast) and 768 (full quality)
        self.short_dim = 256
        self.long_dim = self.full_dim  # 768 for nomic
        
        print(f"Target dimensions: {self.short_dim} (fast), {self.long_dim} (full)")

    
    def load_documents(self, input_path: str = "data/processed_documents.jsonl"):
        """
        Load documents from JSONL file.
        Returns list of documents with text and metadata.
        """
        documents = []
        print(f"Loading documents from {input_path}...")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading"):
                doc = json.loads(line)
                documents.append(doc)
        
        print(f"Loaded {len(documents):,} documents")
        return documents

    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate full-dimension embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (num_texts, full_dim)
        """
        embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            
            # Generate embeddings
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # L2 normalization for cosine similarity
            )
            
            embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        return np.vstack(embeddings)
    
    def truncate_embeddings(
        self, 
        full_embeddings: np.ndarray, 
        target_dim: int
    ) -> np.ndarray:
        """
        Truncate full embeddings to shorter dimension (MRL).
        
        Args:
            full_embeddings: Full dimension embeddings (N, full_dim)
            target_dim: Target dimension to truncate to
            
        Returns:
            Truncated embeddings (N, target_dim)
        """
        if target_dim >= full_embeddings.shape[1]:
            return full_embeddings
        
        # Simply take first N dimensions (this works for MRL models)
        truncated = full_embeddings[:, :target_dim].copy()
        
        # Re-normalize after truncation
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        truncated = truncated / norms
        
        return truncated

    

    def process_all_documents(
        self,
        documents: List[Dict],
        batch_size: int = 32,
        save_every: int = 10000,
        output_dir: str = "data/embeddings"
    ):
        """
        Process all documents and generate embeddings.
        Saves incrementally to handle large datasets.
        
        Args:
            documents: List of document dictionaries
            batch_size: Batch size for encoding
            save_every: Save checkpoint every N documents
            output_dir: Directory to save embeddings
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting embedding generation for {len(documents):,} documents")
        print(f"{'='*60}\n")
        
        # Extract texts and IDs
        texts = [doc['text'] for doc in documents]
        doc_ids = [doc['id'] for doc in documents]
        
        start_time = time.time()
        
        # Generate full embeddings
        print("📊 Generating full embeddings...")
        full_embeddings = self.generate_embeddings_batch(
            texts, 
            batch_size=batch_size,
            show_progress=True
        )
        
        # Truncate to short dimension
        print(f"\n✂️  Truncating to {self.short_dim} dimensions...")
        short_embeddings = self.truncate_embeddings(full_embeddings, self.short_dim)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ Embedding generation complete!")
        print(f"{'='*60}")
        print(f"⏱️  Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"📈 Speed: {len(documents)/elapsed:.2f} docs/second")
        print(f"📊 Full embeddings shape: {full_embeddings.shape}")
        print(f"📊 Short embeddings shape: {short_embeddings.shape}")
        
        # Save embeddings
        self.save_embeddings(
            doc_ids=doc_ids,
            short_embeddings=short_embeddings,
            full_embeddings=full_embeddings,
            output_dir=output_path
        )
        
        return {
            'doc_ids': doc_ids,
            'short_embeddings': short_embeddings,
            'full_embeddings': full_embeddings,
            'metadata': {
                'num_documents': len(documents),
                'short_dim': self.short_dim,
                'full_dim': self.long_dim,
                'model_name': self.model.model_card_data.model_name if hasattr(self.model, 'model_card_data') else "unknown",
                'generation_time': elapsed
            }
        }

    

    def save_embeddings(
        self,
        doc_ids: List[str],
        short_embeddings: np.ndarray,
        full_embeddings: np.ndarray,
        output_dir: Path
    ):
        """
        Save embeddings and metadata to disk.
        Uses memory-efficient numpy format.
        """
        print(f"\n💾 Saving embeddings to {output_dir}...")
        
        # Save short embeddings (256-dim)
        short_path = output_dir / f"embeddings_{self.short_dim}d.npy"
        np.save(short_path, short_embeddings)
        print(f"  ✅ Saved {self.short_dim}d embeddings: {short_path}")
        print(f"     Size: {short_path.stat().st_size / (1024**2):.2f} MB")
        
        # Save full embeddings (768-dim)
        full_path = output_dir / f"embeddings_{self.long_dim}d.npy"
        np.save(full_path, full_embeddings)
        print(f"  ✅ Saved {self.long_dim}d embeddings: {full_path}")
        print(f"     Size: {full_path.stat().st_size / (1024**2):.2f} MB")
        
        # Save document IDs (to map embeddings back to documents)
        ids_path = output_dir / "document_ids.json"
        with open(ids_path, 'w', encoding='utf-8') as f:
            json.dump(doc_ids, f)
        print(f"  ✅ Saved document IDs: {ids_path}")
        
        # Save metadata
        metadata = {
            'num_documents': len(doc_ids),
            'short_dim': self.short_dim,
            'full_dim': self.long_dim,
            'model': "nomic-ai/nomic-embed-text-v1.5",
            'normalization': 'L2',
            'files': {
                'short_embeddings': str(short_path.name),
                'full_embeddings': str(full_path.name),
                'document_ids': str(ids_path.name)
            }
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✅ Saved metadata: {metadata_path}")
        
        print(f"\n✅ All embeddings saved successfully!")



embeddings_dir = Path("data/embeddings")
short_emb_path = embeddings_dir / "embeddings_256d.npy"
full_emb_path = embeddings_dir / "embeddings_768d.npy"


if short_emb_path.exists() and full_emb_path.exists():
        print("⚠️  Embeddings already exist!")
        print(f"   - {short_emb_path}")
        print(f"   - {full_emb_path}")
        response = input("\nDo you want to regenerate? (yes/no): ").strip().lower()
        if response != 'yes':
            print("✅ Using existing embeddings. Skipping generation.")
            exit(0)



generator = EmbeddingGenerator(model_name="nomic-ai/nomic-embed-text-v1.5")
documents = generator.load_documents("data/processed_documents.jsonl")
results = generator.process_all_documents(
        documents,
        batch_size=8,  # Adjust based on your GPU memory
        output_dir="data/embeddings"
    )

print("\n" + "="*60)
print("PART 2 COMPLETE!")
print("="*60)
print(f" Generated embeddings for {len(documents):,} documents")
print(f" Saved to: data/embeddings/")
print(f"\n Files created:")
print(f"   - embeddings_256d.npy  ({results['metadata']['short_dim']} dimensions)")
print(f"   - embeddings_768d.npy  ({results['metadata']['full_dim']} dimensions)")
print(f"   - document_ids.json")
print(f"   - metadata.json")
print("\nReady for Part 3: Building the Vector Index!")