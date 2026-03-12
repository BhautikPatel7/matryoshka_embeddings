import numpy as np
import json
import time
import psutil
import os
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from tqdm import tqdm
import pandas as pd

# Import from Part 3
import sys
sys.path.append('.')
from build_vector_index import VectorSearchEngine

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    mode: str
    query: str
    latency_ms: float
    recall_at_10: float
    memory_mb: float
    top_10_ids: List[str]
    
class VectorSearchBenchmark:
    """
    Comprehensive benchmarking system for vector search.
    Measures quality (Recall@10) and performance (latency, memory).
    """
    
    def __init__(self, search_engine: VectorSearchEngine):
        self.search_engine = search_engine
        self.results = []
        
    def create_ground_truth(self, queries: List[str]) -> Dict[str, List[str]]:
        """
        Create ground truth using Mode 1 (full 768-dim search).
        This serves as our "gold standard" for recall calculation.
        
        Args:
            queries: List of test queries
            
        Returns:
            Dict mapping query → list of top 10 doc IDs
        """
        print("\n📊 Creating ground truth (using Mode 1 - Full 768d)...")
        ground_truth = {}
        
        for query in tqdm(queries, desc="Ground truth"):
            results, _ = self.search_engine.search_mode1_full(query, top_k=10)
            ground_truth[query] = [r.doc_id for r in results]
        
        print(f"✅ Ground truth created for {len(queries)} queries\n")
        return ground_truth
    
    def calculate_recall_at_k(
        self, 
        retrieved_ids: List[str], 
        ground_truth_ids: List[str],
        k: int = 10
    ) -> float:
        """
        Calculate Recall@K metric.
        
        Recall@K = (# of relevant docs retrieved in top K) / K
        
        Args:
            retrieved_ids: IDs returned by search method
            ground_truth_ids: Ground truth IDs (from Mode 1)
            k: Number of results to consider
            
        Returns:
            Recall score (0.0 to 1.0)
        """
        retrieved_set = set(retrieved_ids[:k])
        ground_truth_set = set(ground_truth_ids[:k])
        
        overlap = len(retrieved_set.intersection(ground_truth_set))
        recall = overlap / k
        
        return recall
    
    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        return memory_mb
    
    def benchmark_single_query(
        self,
        query: str,
        mode: str,
        ground_truth_ids: List[str],
        runs: int = 5
    ) -> BenchmarkResult:
        """
        Benchmark a single query with multiple runs for accuracy.
        
        Args:
            query: Search query
            mode: 'mode1', 'mode2', or 'mode3'
            ground_truth_ids: Ground truth doc IDs
            runs: Number of runs to average
            
        Returns:
            BenchmarkResult with averaged metrics
        """
        latencies = []
        memory_before = self.measure_memory_usage()
        
        # Run multiple times for accurate timing
        for _ in range(runs):
            start_time = time.time()
            
            if mode == 'mode1':
                results, _ = self.search_engine.search_mode1_full(query, top_k=10)
            elif mode == 'mode2':
                results, _ = self.search_engine.search_mode2_fast(query, top_k=10)
            elif mode == 'mode3':
                results, _ = self.search_engine.search_mode3_mrl(query, top_k=10, first_stage_k=100)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        memory_after = self.measure_memory_usage()
        
        # Get final results for recall calculation
        retrieved_ids = [r.doc_id for r in results]
        recall = self.calculate_recall_at_k(retrieved_ids, ground_truth_ids, k=10)
        
        # Average latency
        avg_latency = np.mean(latencies)
        memory_used = memory_after - memory_before
        
        return BenchmarkResult(
            mode=mode,
            query=query,
            latency_ms=avg_latency,
            recall_at_10=recall,
            memory_mb=memory_used,
            top_10_ids=retrieved_ids
        )
    
    def run_comprehensive_benchmark(
        self,
        test_queries: List[str],
        runs_per_query: int = 5
    ) -> pd.DataFrame:
        """
        Run comprehensive benchmark across all modes and queries.
        
        Args:
            test_queries: List of test queries
            runs_per_query: Number of runs per query for averaging
            
        Returns:
            DataFrame with all benchmark results
        """
        print("\n" + "="*80)
        print("🚀 STARTING COMPREHENSIVE BENCHMARK")
        print("="*80)
        print(f"Test queries: {len(test_queries)}")
        print(f"Runs per query: {runs_per_query}")
        print(f"Total tests: {len(test_queries) * 3 * runs_per_query}")
        print("="*80 + "\n")
        
        # Create ground truth
        ground_truth = self.create_ground_truth(test_queries)
        
        modes = ['mode1', 'mode2', 'mode3']
        mode_names = {
            'mode1': 'Full 768d',
            'mode2': 'Fast 256d', 
            'mode3': 'MRL Two-Stage'
        }
        
        all_results = []
        
        # Benchmark each mode
        for mode in modes:
            print(f"\n📊 Benchmarking {mode_names[mode]}...")
            
            for query in tqdm(test_queries, desc=f"{mode_names[mode]}"):
                result = self.benchmark_single_query(
                    query=query,
                    mode=mode,
                    ground_truth_ids=ground_truth[query],
                    runs=runs_per_query
                )
                all_results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in all_results])
        
        print("\n" + "="*80)
        print("✅ BENCHMARK COMPLETE!")
        print("="*80)
        
        return df
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics by mode"""
        
        summary = df.groupby('mode').agg({
            'latency_ms': ['mean', 'std', 'min', 'max'],
            'recall_at_10': ['mean', 'std', 'min', 'max'],
            'memory_mb': 'mean'
        }).round(2)
        
        return summary
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str = "data/benchmarks"):
        """
        Create comprehensive visualization charts.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        mode_labels = {
            'mode1': 'Full 768d',
            'mode2': 'Fast 256d',
            'mode3': 'MRL Two-Stage'
        }
        
        # Replace mode names
        df_viz = df.copy()
        df_viz['mode'] = df_viz['mode'].map(mode_labels)
        
        # 1. Latency Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df_viz, x='mode', y='latency_ms', palette='Set2', ax=ax)
        ax.set_title('Latency Comparison Across Search Modes', fontsize=16, fontweight='bold')
        ax.set_xlabel('Search Mode', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path / 'latency_comparison.png'}")
        plt.close()
        
        # 2. Recall@10 Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df_viz, x='mode', y='recall_at_10', palette='Set1', ax=ax)
        ax.set_title('Recall@10 Comparison Across Search Modes', fontsize=16, fontweight='bold')
        ax.set_xlabel('Search Mode', fontsize=12)
        ax.set_ylabel('Recall@10', fontsize=12)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(output_path / 'recall_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path / 'recall_comparison.png'}")
        plt.close()
        
        # 3. Quality vs Speed Tradeoff (MOST IMPORTANT!)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        summary = df_viz.groupby('mode').agg({
            'latency_ms': 'mean',
            'recall_at_10': 'mean'
        }).reset_index()
        
        colors = {'Full 768d': '#FF6B6B', 'Fast 256d': '#4ECDC4', 'MRL Two-Stage': '#95E1D3'}
        
        for _, row in summary.iterrows():
            ax.scatter(
                row['latency_ms'], 
                row['recall_at_10'],
                s=500,
                alpha=0.6,
                color=colors[row['mode']],
                label=row['mode'],
                edgecolors='black',
                linewidth=2
            )
            ax.annotate(
                row['mode'],
                (row['latency_ms'], row['recall_at_10']),
                fontsize=11,
                fontweight='bold',
                ha='center',
                va='bottom',
                xytext=(0, 10),
                textcoords='offset points'
            )
        
        ax.set_xlabel('Average Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Recall@10', fontsize=12, fontweight='bold')
        ax.set_title('Quality vs Speed Tradeoff\n(Top-right is best: High Recall, Low Latency)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / 'quality_vs_speed.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path / 'quality_vs_speed.png'}")
        plt.close()
        
        # 4. Performance Summary Table (as image)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        
        summary_stats = df.groupby('mode').agg({
            'latency_ms': 'mean',
            'recall_at_10': 'mean',
        }).reset_index()
        
        summary_stats['mode'] = summary_stats['mode'].map(mode_labels)
        summary_stats['latency_ms'] = summary_stats['latency_ms'].round(2)
        summary_stats['recall_at_10'] = summary_stats['recall_at_10'].round(4)
        
        # Add speedup column
        baseline_latency = summary_stats[summary_stats['mode'] == 'Full 768d']['latency_ms'].values[0]
        summary_stats['speedup'] = (baseline_latency / summary_stats['latency_ms']).round(2)
        summary_stats['speedup'] = summary_stats['speedup'].astype(str) + 'x'
        
        summary_stats.columns = ['Mode', 'Avg Latency (ms)', 'Avg Recall@10', 'Speedup vs Baseline']
        
        table = ax.table(
            cellText=summary_stats.values,
            colLabels=summary_stats.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.3, 0.25, 0.25, 0.25]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(summary_stats.columns)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        colors_list = ['#FFE5E5', '#E5F9F9', '#E5F9E9']
        for i in range(1, len(summary_stats) + 1):
            for j in range(len(summary_stats.columns)):
                table[(i, j)].set_facecolor(colors_list[i-1])
        
        plt.title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(output_path / 'summary_table.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path / 'summary_table.png'}")
        plt.close()
        
        print(f"\n✅ All visualizations saved to: {output_path}/")
    
    def save_results(self, df: pd.DataFrame, output_dir: str = "data/benchmarks"):
        """Save benchmark results to CSV"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results
        csv_path = output_path / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved detailed results: {csv_path}")
        
        # Save summary statistics
        summary = self.generate_summary_statistics(df)
        summary_path = output_path / "benchmark_summary.csv"
        summary.to_csv(summary_path)
        print(f"✅ Saved summary statistics: {summary_path}")
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate a markdown report"""
        
        mode_labels = {
            'mode1': 'Full 768d',
            'mode2': 'Fast 256d',
            'mode3': 'MRL Two-Stage'
        }
        
        summary = df.groupby('mode').agg({
            'latency_ms': 'mean',
            'recall_at_10': 'mean',
        }).reset_index()
        
        baseline_latency = summary[summary['mode'] == 'mode1']['latency_ms'].values[0]
        
        report = f"""# Vector Search System Benchmark Report

## Executive Summary

Comprehensive performance evaluation of 3 vector search strategies on {len(df['query'].unique())} test queries.

## System Configuration

- **Dataset Size**: {len(self.search_engine.doc_ids):,} documents
- **Short Embeddings**: 256 dimensions
- **Full Embeddings**: 768 dimensions
- **Test Queries**: {len(df['query'].unique())}
- **Runs per Query**: {len(df[df['query'] == df['query'].iloc[0]]) // 3}

## Results Summary

| Mode | Avg Latency (ms) | Avg Recall@10 | Speedup |
|------|------------------|---------------|---------|
"""
        
        for _, row in summary.iterrows():
            mode_name = mode_labels[row['mode']]
            speedup = baseline_latency / row['latency_ms']
            report += f"| {mode_name} | {row['latency_ms']:.2f} | {row['recall_at_10']:.4f} | {speedup:.2f}x |\n"
        
        report += f"""
## Key Findings

### 🚀 Performance Improvements

"""
        
        mode2_speedup = baseline_latency / summary[summary['mode'] == 'mode2']['latency_ms'].values[0]
        mode3_speedup = baseline_latency / summary[summary['mode'] == 'mode3']['latency_ms'].values[0]
        
        mode2_recall = summary[summary['mode'] == 'mode2']['recall_at_10'].values[0]
        mode3_recall = summary[summary['mode'] == 'mode3']['recall_at_10'].values[0]
        
        report += f"""- **Fast 256d Mode**: {mode2_speedup:.2f}x faster with {mode2_recall:.2%} recall
- **MRL Two-Stage Mode**: {mode3_speedup:.2f}x faster with {mode3_recall:.2%} recall

### 📊 Recommendation

**Use Mode 3 (MRL Two-Stage)** for production:
- Achieves {mode3_speedup:.2f}x speedup over baseline
- Maintains {mode3_recall:.2%} recall quality
- Best quality-speed tradeoff

## Visualizations

See the following charts in `data/benchmarks/`:
- `latency_comparison.png` - Latency distribution
- `recall_comparison.png` - Recall@10 distribution
- `quality_vs_speed.png` - Quality vs Speed tradeoff
- `summary_table.png` - Performance summary table

---
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    # Initialize search engine
    print("🚀 Initializing Search Engine...")
    search_engine = VectorSearchEngine()
    
    # Create benchmark suite
    benchmark = VectorSearchBenchmark(search_engine)
    
    # Define test queries
    test_queries = [
        "How to reverse a string in Python?",
        "JavaScript async await tutorial",
        "SQL join multiple tables",
        "React hooks useState example",
        "Python list comprehension",
        "CSS flexbox layout",
        "Git merge vs rebase",
        "Docker container tutorial",
        "REST API best practices",
        "Binary search algorithm",
    ]
    
    print(f"\n📋 Test queries: {len(test_queries)}")
    
    # Run comprehensive benchmark
    results_df = benchmark.run_comprehensive_benchmark(
        test_queries=test_queries,
        runs_per_query=5  # Run each query 5 times for accuracy
    )
    
    # Generate summary
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    summary = benchmark.generate_summary_statistics(results_df)
    print(summary)
    
    # Create visualizations
    print("\n📈 Generating visualizations...")
    benchmark.create_visualizations(results_df)
    
    # Save results
    print("\n💾 Saving results...")
    benchmark.save_results(results_df)
    
    # Generate report
    report = benchmark.generate_report(results_df)
    report_path = Path("data/benchmarks/REPORT.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Saved report: {report_path}")
    
    print("\n" + "="*80)
    print("🎉 PART 4 COMPLETE!")
    print("="*80)
    print(f"\n📁 All results saved to: data/benchmarks/")
    print(f"   - benchmark_results.csv (detailed data)")
    print(f"   - benchmark_summary.csv (statistics)")
    print(f"   - REPORT.md (markdown report)")
    print(f"   - *.png (visualizations)")
    print("\n🚀 Next: Part 5 - Scale Testing & Advanced Optimizations!")