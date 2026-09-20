"""
Evaluation Runner Script for Generation Quality & Latency in Traffic Law RAG (Prompt 8.5).
Executes live evaluation against local Ollama Qwen and BGE-M3 Dense / BM25 / Hybrid retrievers.

Key features:
1. Fully Resumable: saves each completed case immediately; skips already completed cases unless --rerun.
2. Cold vs Warm Latency Diagnostic: 1 cold start request + 3 warm requests, saved to cold_warm_benchmark.json.
3. Context Size Diagnostic: top_k=3 vs top_k=5 comparison, saved to context_benchmark.json.
4. Outputs generation_results.json, generation_summary.json, and generation_summary.csv in evaluation/results_generation/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bge_embedder import BGEEmbedder
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.qwen_client import QwenClient
from src.generation.rag_generator import RAGGenerator
from src.evaluation.generation_evaluator import GenerationEvaluator

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "generation_eval.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results_generation"
CORPUS_PATH = PROJECT_ROOT / "data" / "chunks" / "legal_chunks.json"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"


def print_separator(title: str = "") -> None:
    print("\n" + "=" * 80)
    if title:
        print(f" {title.upper()} ".center(80, "="))
        print("=" * 80)


def load_existing_results(results_file: Path) -> dict[str, dict[str, Any]]:
    """Load previously completed case results to support resuming."""
    if not results_file.exists():
        return {}
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return {item["id"]: item for item in data if "id" in item}
    except Exception as e:
        print(f"Warning: Could not read existing results from {results_file}: {e}")
    return {}


def save_results(results_file: Path, results_map: dict[str, dict[str, Any]]) -> None:
    """Atomically save per-case evaluation results to JSON."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = results_file.with_suffix(".tmp")
    items = list(results_map.values())
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    temp_file.replace(results_file)


def export_csv(csv_file: Path, results: list[dict[str, Any]]) -> None:
    """Export summary metrics to CSV."""
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "query",
        "expected_status",
        "actual_status",
        "status_correct",
        "retrieval_support_complete",
        "completeness_status",
        "completeness_pass",
        "citation_valid",
        "citation_coverage_score",
        "groundedness_pass",
        "manual_review_required",
        "generation_ms",
        "total_ms",
        "notes",
    ]
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "id": r.get("id"),
                "category": r.get("category"),
                "query": r.get("query"),
                "expected_status": r.get("expected_status"),
                "actual_status": r.get("actual_status"),
                "status_correct": r.get("status_correct"),
                "retrieval_support_complete": r.get("retrieval_support_complete"),
                "completeness_status": r.get("completeness_status"),
                "completeness_pass": r.get("completeness_pass"),
                "citation_valid": r.get("citation_valid"),
                "citation_coverage_score": r.get("citation_coverage_score"),
                "groundedness_pass": r.get("groundedness_pass"),
                "manual_review_required": r.get("manual_review_required"),
                "generation_ms": r.get("latency", {}).get("generation_ms", 0.0),
                "total_ms": r.get("latency", {}).get("total_ms", 0.0),
                "notes": " | ".join(r.get("notes", [])),
            })


def run_cold_warm_benchmark(
    generator: RAGGenerator,
    output_dir: Path,
    benchmark_query: str = "Đèn đỏ có được đi không?",
    force: bool = False,
) -> dict[str, Any]:
    """Execute 1 cold request followed by 3 warm requests to diagnose model loading latency."""
    bench_file = output_dir / "cold_warm_benchmark.json"
    if bench_file.exists() and not force:
        try:
            with open(bench_file, "r", encoding="utf-8") as f:
                print(f"[Info] Loading existing Cold vs Warm benchmark from {bench_file}")
                return json.load(f)
        except Exception:
            pass

    print_separator("COLD VS WARM LATENCY BENCHMARK")
    print(f"Benchmark query: '{benchmark_query}'")

    # 1. Cold request
    print("\nExecuting Cold Request (may include initial model load / disk paging)...")
    r_cold = generator.generate(benchmark_query)
    cold_total_ms = r_cold["latency"]["total_ms"]
    cold_gen_ms = r_cold["latency"]["generation_ms"]
    cold_load_ns = r_cold.get("qwen_metrics", {}).get("load_duration", 0)
    cold_load_ms = round(cold_load_ns / 1e6, 2) if cold_load_ns else 0.0
    print(f"Cold Request Finished: Total={cold_total_ms:.2f}ms | Gen={cold_gen_ms:.2f}ms | Load={cold_load_ms:.2f}ms")

    # 2. Warm requests (3 runs)
    warm_runs = []
    for i in range(1, 4):
        print(f"Executing Warm Request {i}/3...")
        r_warm = generator.generate(benchmark_query)
        w_tot = r_warm["latency"]["total_ms"]
        w_gen = r_warm["latency"]["generation_ms"]
        w_load_ns = r_warm.get("qwen_metrics", {}).get("load_duration", 0)
        w_load_ms = round(w_load_ns / 1e6, 2) if w_load_ns else 0.0
        warm_runs.append({
            "run": i,
            "total_ms": w_tot,
            "generation_ms": w_gen,
            "load_duration_ms": w_load_ms,
        })
        print(f"  Run {i}: Total={w_tot:.2f}ms | Gen={w_gen:.2f}ms | Load={w_load_ms:.2f}ms")

    warm_avg_total = round(sum(r["total_ms"] for r in warm_runs) / len(warm_runs), 2)
    warm_avg_gen = round(sum(r["generation_ms"] for r in warm_runs) / len(warm_runs), 2)

    bench_data = {
        "benchmark_query": benchmark_query,
        "cold_request": {
            "total_ms": cold_total_ms,
            "generation_ms": cold_gen_ms,
            "load_duration_ms": cold_load_ms,
        },
        "warm_runs": warm_runs,
        "warm_avg_total_ms": warm_avg_total,
        "warm_avg_gen_ms": warm_avg_gen,
        "cold_to_warm_total_ratio": round(cold_total_ms / warm_avg_total, 2) if warm_avg_total else 1.0,
    }

    with open(bench_file, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, ensure_ascii=False, indent=2)

    return bench_data


def run_context_size_benchmark(
    generator: RAGGenerator,
    output_dir: Path,
    benchmark_query: str = "Nồng độ cồn xe máy phạt bao nhiêu?",
    force: bool = False,
) -> dict[str, Any]:
    """Compare top_k=3 vs top_k=5 for latency and completeness."""
    bench_file = output_dir / "context_benchmark.json"
    if bench_file.exists() and not force:
        try:
            with open(bench_file, "r", encoding="utf-8") as f:
                print(f"[Info] Loading existing Context Size benchmark from {bench_file}")
                return json.load(f)
        except Exception:
            pass

    print_separator("CONTEXT SIZE BENCHMARK (top_k=3 vs top_k=5)")
    print(f"Query: '{benchmark_query}'")

    # top_k = 3
    print("\nRunning with top_k=3...")
    r3 = generator.generate(benchmark_query, top_k=3)
    k3_data = {
        "top_k": 3,
        "retrieved_chunk_count": len(r3.get("retrieved_chunks", [])),
        "context_chars": sum(len(c.get("content", "")) for c in r3.get("retrieved_chunks", [])),
        "answer_chars": len(r3.get("answer") or ""),
        "total_ms": r3["latency"]["total_ms"],
        "generation_ms": r3["latency"]["generation_ms"],
        "retrieval_ms": r3["latency"]["retrieval_ms"],
        "answer_snippet": (r3.get("answer") or "")[:200],
    }
    print(f"top_k=3 -> Chunks={k3_data['retrieved_chunk_count']}, Total={k3_data['total_ms']:.2f}ms, Gen={k3_data['generation_ms']:.2f}ms")

    # top_k = 5
    print("\nRunning with top_k=5...")
    r5 = generator.generate(benchmark_query, top_k=5)
    k5_data = {
        "top_k": 5,
        "retrieved_chunk_count": len(r5.get("retrieved_chunks", [])),
        "context_chars": sum(len(c.get("content", "")) for c in r5.get("retrieved_chunks", [])),
        "answer_chars": len(r5.get("answer") or ""),
        "total_ms": r5["latency"]["total_ms"],
        "generation_ms": r5["latency"]["generation_ms"],
        "retrieval_ms": r5["latency"]["retrieval_ms"],
        "answer_snippet": (r5.get("answer") or "")[:200],
    }
    print(f"top_k=5 -> Chunks={k5_data['retrieved_chunk_count']}, Total={k5_data['total_ms']:.2f}ms, Gen={k5_data['generation_ms']:.2f}ms")

    bench_data = {
        "benchmark_query": benchmark_query,
        "top_k_3": k3_data,
        "top_k_5": k5_data,
        "latency_diff_ms": round(k5_data["total_ms"] - k3_data["total_ms"], 2),
        "context_chars_diff": k5_data["context_chars"] - k3_data["context_chars"],
    }

    with open(bench_file, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, ensure_ascii=False, indent=2)

    return bench_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Generation Quality and Latency (Prompt 8.5)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to evaluation dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save evaluation results")
    parser.add_argument("--rerun", action="store_true", help="Force rerun already evaluated cases")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip cold/warm and context benchmarks")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "generation_results.json"
    summary_file = output_dir / "generation_summary.json"
    csv_file = output_dir / "generation_summary.csv"

    print_separator("PROMPT 8.5: GENERATION QUALITY & LATENCY EVALUATION")
    print(f"Dataset File   : {args.dataset}")
    print(f"Output Dir     : {output_dir}")
    print(f"Rerun Mode     : {args.rerun}")

    # 1. Load dataset
    with open(args.dataset, "r", encoding="utf-8") as f:
        cases: list[dict[str, Any]] = json.load(f)
    print(f"Loaded {len(cases)} evaluation cases.")

    # 2. Initialize RAG Components
    print("\n[1/3] Initializing Retrievers & Local Qwen Client...")
    qwen_client = QwenClient()
    embedder = BGEEmbedder()
    dense_store = FAISSStore.load(VECTOR_STORE_DIR, embedder=embedder)
    bm25_retriever = BM25Retriever.from_json(CORPUS_PATH)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_store,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
        corpus_path=CORPUS_PATH,
        validate_consistency=True,
    )
    generator = RAGGenerator(
        retriever={
            "dense": dense_store,
            "bm25": bm25_retriever,
            "hybrid": hybrid_retriever,
        },
        qwen_client=qwen_client,
        default_retrieval_mode="dense",
        top_k=5,
    )
    evaluator = GenerationEvaluator()
    print("Components ready.")

    # 3. Optional Diagnostics: Cold vs Warm and Context Size Benchmarks
    if not args.skip_benchmarks:
        cold_warm_results = run_cold_warm_benchmark(generator, output_dir, force=args.rerun)
        context_bench_results = run_context_size_benchmark(generator, output_dir, force=args.rerun)
    else:
        print("\n[Skipping Cold/Warm & Context Benchmarks as requested]")

    # 4. Resumable Per-Case Evaluation
    print_separator("EXECUTING EVALUATION DATASET")
    completed_map = load_existing_results(results_file) if not args.rerun else {}
    print(f"Previously completed cases found: {len(completed_map)} / {len(cases)}")

    case_order_ids = [c["id"] for c in cases]
    for idx, case in enumerate(cases, start=1):
        cid = case["id"]
        q = case["query"]
        exp_status = case["expected_status"]

        if cid in completed_map and not args.rerun:
            prev = completed_map[cid]
            status_str = "CORRECT" if prev.get("status_correct") else "FAIL"
            comp_str = prev.get("completeness_status", "N/A")
            print(f"[{idx}/{len(cases)}] {cid} (Cached): Status={prev.get('actual_status')} ({status_str}) | Comp={comp_str} | Lat={prev.get('latency', {}).get('total_ms', 0):.1f}ms")
            continue

        print(f"\n[{idx}/{len(cases)}] Evaluating {cid} ({case.get('category')}): '{q}'")
        t_start = time.perf_counter()
        case_res = evaluator.evaluate_case(case, generator)
        elapsed_case = (time.perf_counter() - t_start) * 1000.0

        # Save immediately to disk (Resumable guarantee)
        completed_map[cid] = case_res
        save_results(results_file, completed_map)

        # Print per-case feedback
        status_ok = "PASS" if case_res["status_correct"] else "FAIL"
        comp_status = case_res.get("completeness_status")
        grounded_ok = "PASS" if case_res["groundedness_pass"] else "FAIL"
        cit_cov = case_res.get("citation_coverage_score", 0.0)
        tot_lat = case_res.get("latency", {}).get("total_ms", 0.0)
        gen_lat = case_res.get("latency", {}).get("generation_ms", 0.0)

        print(f"  Result: Status={case_res['actual_status']} ({status_ok}) | Completeness={comp_status} | Groundedness={grounded_ok} | CitCov={cit_cov * 100:.0f}%")
        print(f"  Latency: Gen={gen_lat:.1f}ms | Total={tot_lat:.1f}ms (Eval run: {elapsed_case:.1f}ms)")
        if case_res["notes"]:
            print(f"  Notes: {case_res['notes']}")

    # Order results to match dataset
    ordered_results = [completed_map[cid] for cid in case_order_ids if cid in completed_map]

    # Export CSV
    export_csv(csv_file, ordered_results)

    # 5. Summary Metrics Aggregation
    summary = evaluator.aggregate_summary(ordered_results)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 6. Print Execution Summary Table
    print_separator("EVALUATION SUMMARY REPORT")
    print(f"{'Metric':<42} | {'Value':<15}")
    print("-" * 60)
    print(f"{'Total Evaluation Cases':<42} | {summary.get('total_cases', 0)}")
    print(f"{'Status Accuracy':<42} | {summary.get('status_accuracy', 0) * 100:.2f}%")
    print(f"{'Clarification Required Accuracy':<42} | {summary.get('clarification_required_accuracy', 0) * 100:.2f}%")
    print(f"{'Clarification Not Overtriggered':<42} | {summary.get('clarification_not_overtriggered_accuracy', 0) * 100:.2f}%")
    print(f"{'Multi-turn Resolution Accuracy':<42} | {summary.get('multi_turn_resolution_accuracy', 0) * 100:.2f}%")
    print(f"{'Retrieval Support Complete Rate':<42} | {summary.get('retrieval_support_rate', 0) * 100:.2f}%")
    print(f"{'Completeness Pass Rate (Evaluable)':<42} | {summary.get('completeness_pass_rate', 0) * 100:.2f}%")
    print(f"{'Citation Correctness Rate':<42} | {summary.get('citation_correctness_rate', 0) * 100:.2f}%")
    print(f"{'Citation Coverage (Average Score)':<42} | {summary.get('citation_coverage_average', 0) * 100:.2f}%")
    print(f"{'Groundedness Pass Rate':<42} | {summary.get('groundedness_pass_rate', 0) * 100:.2f}%")
    print(f"{'Refusal Correctness Rate':<42} | {summary.get('refusal_correctness_rate', 0) * 100:.2f}%")
    print(f"{'Prompt Injection Defense Rate':<42} | {summary.get('prompt_injection_defense_rate', 0) * 100:.2f}%")
    print(f"{'Manual Review Required Cases':<42} | {summary.get('manual_review_count', 0)}")
    print("-" * 60)

    gen_l = summary.get("latency", {}).get("generation_ms", {})
    tot_l = summary.get("latency", {}).get("total_ms", {})
    print(f"{'Generation Latency P50 / Avg / P95 / Max':<42} | {gen_l.get('p50', 0):.0f} / {gen_l.get('avg', 0):.0f} / {gen_l.get('p95', 0):.0f} / {gen_l.get('max', 0):.0f} ms")
    print(f"{'Total Pipeline Latency P50 / Avg / P95 / Max':<42} | {tot_l.get('p50', 0):.0f} / {tot_l.get('avg', 0):.0f} / {tot_l.get('p95', 0):.0f} / {tot_l.get('max', 0):.0f} ms")

    out_p = summary.get("output_profiling", {})
    print(f"{'Avg Context Chars / Answer Chars':<42} | {out_p.get('avg_context_chars', 0)} / {out_p.get('avg_answer_chars', 0)}")
    print(f"{'Avg Output Tokens (eval_count)':<42} | {out_p.get('avg_output_tokens', 0)}")
    print("-" * 60)

    # Bottleneck diagnosis
    if tot_l.get("avg", 0) > 0:
        gen_ratio = gen_l.get("avg", 0) / tot_l.get("avg", 1) * 100
        print(f"BOTTLENECK DIAGNOSIS: Generation accounts for {gen_ratio:.1f}% of total pipeline latency.")
        if gen_ratio > 80.0:
            print(">> Qwen LLM inference is conclusively confirmed as the primary pipeline bottleneck.")

    print(f"\nArtifacts saved in:\n  - {results_file}\n  - {summary_file}\n  - {csv_file}")
    print("\nPrompt 8.5 Evaluation Complete.")


if __name__ == "__main__":
    main()
