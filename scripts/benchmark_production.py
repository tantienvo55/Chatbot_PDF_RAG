"""
Production Benchmark Script (Prompt 8.6).
Performs fair comparison between:
- Config A: Cold Baseline (unwarmed load)
- Config B: Warm Baseline (top_k=5, non-stream, unlimited num_predict)
- Config C: Warm Production Candidate (streaming, dynamic top-k, num_predict=384, keep_alive=10m)

Measures TTFT, generation latency, total latency, context characters, answer length,
eval_count tokens, and verifies the Quality Gate (retrieval support integrity, groundedness, citations).
Outputs to evaluation/results_production/.
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
from src.generation.qwen_client import QwenClient, warm_up_model
from src.generation.rag_generator import RAGGenerator
from src.evaluation.generation_evaluator import GenerationEvaluator

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results_production"
CORPUS_PATH = PROJECT_ROOT / "data" / "chunks" / "legal_chunks.json"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

# Representative 8-query benchmark set covering all core categories
BENCHMARK_QUERIES = [
    {
        "id": "P01",
        "query": "Đèn đỏ có được đi không?",
        "category": "general_rule",
        "expected_status": "ANSWER",
        "expected_chunks": ["L36_DIEU11_KHOAN4_DIEMC"],
        "required_topics": ["cấm đi", "dừng lại"],
    },
    {
        "id": "P02",
        "query": "Nồng độ cồn xe máy phạt bao nhiêu?",
        "category": "multi_bracket_penalty",
        "expected_status": "ANSWER",
        "expected_chunks": ["ND168_DIEU7_KHOAN6_DIEMA", "ND168_DIEU7_KHOAN8_DIEMB", "ND168_DIEU7_KHOAN9_DIEMD"],
        "required_topics": ["2.000.000", "6.000.000", "10.000.000"],
    },
    {
        "id": "P03",
        "query": "Không đội mũ bảo hiểm xe máy phạt bao nhiêu?",
        "category": "single_penalty",
        "expected_status": "ANSWER",
        "expected_chunks": ["ND168_DIEU7_KHOAN2_DIEMH"],
        "required_topics": ["400.000", "600.000"],
    },
    {
        "id": "P04",
        "query": "Giấy phép lái xe bị trừ điểm trong trường hợp nào?",
        "category": "legal_conditions",
        "expected_status": "ANSWER",
        "expected_chunks": ["L36_DIEU58_KHOAN1", "L36_DIEU58_KHOAN2"],
        "required_topics": ["trừ điểm", "vi phạm"],
    },
    {
        "id": "P05",
        "query": "Thế nào là dừng xe và đỗ xe?",
        "category": "definition",
        "expected_status": "ANSWER",
        "expected_chunks": ["L36_DIEU18_KHOAN1", "L36_DIEU18_KHOAN2"],
        "required_topics": ["đứng yên tạm thời", "đứng yên"],
    },
    {
        "id": "P06",
        "query": "Nồng độ cồn phạt bao nhiêu?",
        "category": "clarification_required",
        "expected_status": "CLARIFY",
        "expected_chunks": [],
        "required_topics": [],
    },
    {
        "id": "P07",
        "query": "Thuế thu nhập cá nhân tính thế nào?",
        "category": "out_of_scope",
        "expected_status": "OUT_OF_SCOPE",
        "expected_chunks": [],
        "required_topics": [],
    },
    {
        "id": "P08",
        "query": "Thời gian bảo hành của thiết bị bay không người lái flycam theo quy định hàng không là bao lâu?",
        "category": "insufficient_context",
        "expected_status": "INSUFFICIENT_CONTEXT",
        "expected_chunks": [],
        "required_topics": [],
    },
]


def print_separator(title: str = "") -> None:
    print("\n" + "=" * 80)
    if title:
        print(f" {title.upper()} ".center(80, "="))
        print("=" * 80)


def execute_streaming_run(generator: RAGGenerator, query: str) -> tuple[dict[str, Any], list[str]]:
    """Execute generator.generate_stream and collect all streamed token events and final payload."""
    streamed_tokens: list[str] = []
    final_response: dict[str, Any] = {}
    for event in generator.generate_stream(query):
        if event.get("type") == "token":
            streamed_tokens.append(event.get("text", ""))
        elif event.get("type") == "final":
            final_response = event.get("response", {})
    return final_response, streamed_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Production Benchmark (Prompt 8.6)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-predict", type=int, default=384, help="num_predict for candidate")
    parser.add_argument("--rerun", action="store_true", help="Force rerun all benchmarks")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_file = output_dir / "production_benchmark.json"
    summary_file = output_dir / "production_summary.json"
    csv_file = output_dir / "production_summary.csv"

    print_separator("PROMPT 8.6: PRODUCTION BENCHMARK")
    print(f"Output Directory  : {output_dir}")
    print(f"Num Predict Candidate : {args.num_predict}")

    # Initialize Retrievers
    print("\n[1/3] Initializing Retrieval Stores...")
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
    retrievers = {
        "dense": dense_store,
        "bm25": bm25_retriever,
        "hybrid": hybrid_retriever,
    }
    evaluator = GenerationEvaluator()

    cold_chk = output_dir / "checkpoint_cold_baseline.json"
    base_chk = output_dir / "checkpoint_baseline_results.json"
    cand_chk = output_dir / "checkpoint_candidate_results.json"

    # =========================================================================
    # 1. COLD BASELINE RUN (Request 1 on unwarmed model)
    # =========================================================================
    print_separator("CONFIG A: COLD BASELINE")
    if not args.rerun and cold_chk.exists():
        print(f"Loading cached Cold Baseline from {cold_chk}...")
        with open(cold_chk, "r", encoding="utf-8") as f:
            cold_record = json.load(f)
    else:
        cold_qwen = QwenClient(timeout=300.0, num_predict=None, keep_alive="10m")
        cold_gen = RAGGenerator(
            retriever=retrievers,
            qwen_client=cold_qwen,
            default_retrieval_mode="dense",
            top_k=5,
            enable_dynamic_top_k=False,
        )
        cold_query = BENCHMARK_QUERIES[0]["query"]
        print(f"Executing Cold Baseline query: '{cold_query}'...")
        cold_res = cold_gen.generate(cold_query)
        cold_load_ns = cold_res.get("qwen_metrics", {}).get("load_duration", 0)
        cold_load_ms = round(cold_load_ns / 1e6, 2) if cold_load_ns else 0.0
        cold_record = {
            "query": cold_query,
            "total_ms": cold_res["latency"]["total_ms"],
            "generation_ms": cold_res["latency"]["generation_ms"],
            "load_duration_ms": cold_load_ms,
            "eval_count": cold_res.get("qwen_metrics", {}).get("eval_count", 0),
        }
        with open(cold_chk, "w", encoding="utf-8") as f:
            json.dump(cold_record, f, ensure_ascii=False, indent=2)

    print(f"Cold Baseline -> Total: {cold_record['total_ms']:.1f}ms | Gen: {cold_record['generation_ms']:.1f}ms | Model Load: {cold_record['load_duration_ms']:.1f}ms")

    # =========================================================================
    # 2. WARM BASELINE RUN (top_k=5, non-stream, unlimited num_predict)
    # =========================================================================
    print_separator("CONFIG B: WARM BASELINE")
    baseline_results: list[dict[str, Any]] = []
    if not args.rerun and base_chk.exists():
        print(f"Loading cached Warm Baseline results from {base_chk}...")
        with open(base_chk, "r", encoding="utf-8") as f:
            baseline_results = json.load(f)
        for r in baseline_results:
            print(f"  [Loaded {r['id']}] Status={r['actual_status']} | Gen={r['generation_ms']:.1f}ms | Tot={r['total_ms']:.1f}ms | EvalCount={r['eval_count']}")

    if len(baseline_results) < len(BENCHMARK_QUERIES):
        warm_base_qwen = QwenClient(timeout=300.0, num_predict=None, keep_alive="10m")
        warm_base_gen = RAGGenerator(
            retriever=retrievers,
            qwen_client=warm_base_qwen,
            default_retrieval_mode="dense",
            top_k=5,
            enable_dynamic_top_k=False,
        )
        done_qids = {r["id"] for r in baseline_results}

        for item in BENCHMARK_QUERIES:
            qid = item["id"]
            if qid in done_qids:
                continue
            q = item["query"]
            print(f"\n[Baseline {qid}] '{q}'...")
            r = warm_base_gen.generate(q)

            # Retrieval support quality check
            retrieved_ids = {c["chunk_id"] for c in r.get("retrieved_chunks", []) if c.get("chunk_id")}
            exp_chunks = item["expected_chunks"]
            support_complete = True
            missing_support = []
            if item["expected_status"] == "ANSWER" and exp_chunks:
                missing_support = [cid for cid in exp_chunks if cid not in retrieved_ids]
                support_complete = (len(missing_support) == 0)

            record = {
                "id": qid,
                "query": q,
                "category": item["category"],
                "expected_status": item["expected_status"],
                "actual_status": r["status"],
                "status_correct": (r["status"] == item["expected_status"]),
                "retrieval_support_complete": support_complete,
                "missing_support": missing_support,
                "retrieved_chunk_count": len(r.get("retrieved_chunks", [])),
                "context_chars": sum(len(c.get("content", "")) for c in r.get("retrieved_chunks", [])),
                "answer_chars": len(r.get("answer") or ""),
                "generation_ms": r["latency"]["generation_ms"],
                "total_ms": r["latency"]["total_ms"],
                "load_duration_ms": round(r.get("qwen_metrics", {}).get("load_duration", 0) / 1e6, 2),
                "eval_count": r.get("qwen_metrics", {}).get("eval_count", 0),
                "ttft_ms": None,
                "answer_snippet": (r.get("answer") or "")[:150],
            }
            baseline_results.append(record)
            with open(base_chk, "w", encoding="utf-8") as f:
                json.dump(baseline_results, f, ensure_ascii=False, indent=2)
            print(f"  Result: Status={record['actual_status']} | Gen={record['generation_ms']:.1f}ms | Tot={record['total_ms']:.1f}ms | EvalCount={record['eval_count']}")

    # =========================================================================
    # 3. WARM PRODUCTION CANDIDATE (streaming, dynamic top-k, num_predict=384)
    # =========================================================================
    print_separator(f"CONFIG C: WARM PRODUCTION CANDIDATE (num_predict={args.num_predict})")
    prod_qwen = QwenClient(
        timeout=300.0,
        num_predict=args.num_predict,
        keep_alive="10m",
    )
    prod_gen = RAGGenerator(
        retriever=retrievers,
        qwen_client=prod_qwen,
        default_retrieval_mode="dense",
        top_k=5,
        enable_dynamic_top_k=True,  # Dynamic top-k enabled
    )

    candidate_results: list[dict[str, Any]] = []
    if not args.rerun and cand_chk.exists():
        print(f"Loading cached Candidate results from {cand_chk}...")
        with open(cand_chk, "r", encoding="utf-8") as f:
            candidate_results = json.load(f)
        for r in candidate_results:
            ttft_str = f"{r['ttft_ms']:.1f}ms" if r.get('ttft_ms') is not None else "N/A"
            print(f"  [Loaded {r['id']}] Status={r['actual_status']} | TTFT={ttft_str} | Gen={r['generation_ms']:.1f}ms | Tot={r['total_ms']:.1f}ms | EvalCount={r['eval_count']}")

    quality_regressions: list[str] = []
    done_cand_qids = {r["id"] for r in candidate_results}

    for idx, item in enumerate(BENCHMARK_QUERIES):
        qid = item["id"]
        if qid in done_cand_qids:
            continue
        q = item["query"]
        print(f"\n[Candidate {qid}] '{q}'...")

        final_r, streamed_tokens = execute_streaming_run(prod_gen, q)

        # Invariant check: final answer == concatenated streamed tokens
        tokens_concat = "".join(streamed_tokens)
        if len(streamed_tokens) > 0:
            assert final_r["answer"] == tokens_concat, f"Invariant violated for {qid}: final_r['answer'] != tokens_concat"
        else:
            assert final_r["status"] in ("CLARIFY", "OUT_OF_SCOPE", "INSUFFICIENT_CONTEXT"), f"Empty tokens for unexpected status {final_r['status']} in {qid}!"

        retrieved_ids = {c["chunk_id"] for c in final_r.get("retrieved_chunks", []) if c.get("chunk_id")}
        exp_chunks = item["expected_chunks"]
        support_complete = True
        missing_support = []
        if item["expected_status"] == "ANSWER" and exp_chunks:
            missing_support = [cid for cid in exp_chunks if cid not in retrieved_ids]
            support_complete = (len(missing_support) == 0)

        # Check retrieval support regression vs baseline (Adjustment 4)
        base_rec = baseline_results[idx]
        support_regressed = False
        if base_rec["retrieval_support_complete"] and not support_complete:
            support_regressed = True
            quality_regressions.append(f"{qid}: QUALITY REGRESSION — RETRIEVAL/CONTEXT SUPPORT (lost chunks: {missing_support})")

        # Groundedness & citation checks
        citations = final_r.get("citations", [])
        cit_valid, _ = evaluator.check_citation_correctness(citations, final_r.get("retrieved_chunks", []))

        record = {
            "id": qid,
            "query": q,
            "category": item["category"],
            "expected_status": item["expected_status"],
            "actual_status": final_r["status"],
            "status_correct": (final_r["status"] == item["expected_status"]),
            "retrieval_support_complete": support_complete,
            "missing_support": missing_support,
            "support_regressed": support_regressed,
            "retrieved_chunk_count": len(final_r.get("retrieved_chunks", [])),
            "context_chars": sum(len(c.get("content", "")) for c in final_r.get("retrieved_chunks", [])),
            "answer_chars": len(final_r.get("answer") or ""),
            "generation_ms": final_r["latency"]["generation_ms"],
            "total_ms": final_r["latency"]["total_ms"],
            "load_duration_ms": round(final_r.get("qwen_metrics", {}).get("load_duration", 0) / 1e6, 2),
            "eval_count": final_r.get("qwen_metrics", {}).get("eval_count", 0),
            "ttft_ms": final_r["latency"].get("ttft_ms"),
            "citation_valid": cit_valid,
            "answer_snippet": (final_r.get("answer") or "")[:150],
        }
        candidate_results.append(record)
        with open(cand_chk, "w", encoding="utf-8") as f:
            json.dump(candidate_results, f, ensure_ascii=False, indent=2)
        ttft_str = f"{record['ttft_ms']:.1f}ms" if record['ttft_ms'] is not None else "N/A"
        print(f"  Result: Status={record['actual_status']} | TTFT={ttft_str} | Gen={record['generation_ms']:.1f}ms | Tot={record['total_ms']:.1f}ms | EvalCount={record['eval_count']}")

    # Check regressions across all candidate results vs baseline
    for idx, c_rec in enumerate(candidate_results):
        b_rec = baseline_results[idx]
        if b_rec["retrieval_support_complete"] and not c_rec["retrieval_support_complete"]:
            reg_msg = f"{c_rec['id']}: QUALITY REGRESSION — RETRIEVAL/CONTEXT SUPPORT (lost chunks: {c_rec['missing_support']})"
            if reg_msg not in quality_regressions:
                quality_regressions.append(reg_msg)

    # =========================================================================
    # 4. SUMMARY COMPARISON & QUALITY GATE
    # =========================================================================
    def calc_stats(records: list[dict[str, Any]], field: str) -> dict[str, float]:
        vals = [r[field] for r in records if r.get(field) is not None and r[field] > 0]
        if not vals:
            return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        vals.sort()
        n = len(vals)
        return {
            "avg": round(sum(vals) / n, 1),
            "p50": round(vals[int(0.5 * n)], 1),
            "p95": round(vals[int(0.95 * n) if n > 1 else -1], 1),
            "max": round(vals[-1], 1),
        }

    base_gen_stats = calc_stats(baseline_results, "generation_ms")
    base_tot_stats = calc_stats(baseline_results, "total_ms")
    cand_gen_stats = calc_stats(candidate_results, "generation_ms")
    cand_tot_stats = calc_stats(candidate_results, "total_ms")
    cand_ttft_stats = calc_stats(candidate_results, "ttft_ms")

    # Quality Gate checks
    base_status_acc = sum(1 for r in baseline_results if r["status_correct"]) / len(baseline_results)
    cand_status_acc = sum(1 for r in candidate_results if r["status_correct"]) / len(candidate_results)
    all_citations_valid = all(r["citation_valid"] for r in candidate_results)
    no_support_regression = (len(quality_regressions) == 0)

    quality_gate_passed = (
        cand_status_acc >= base_status_acc
        and all_citations_valid
        and no_support_regression
    )

    summary_data = {
        "quality_gate_passed": quality_gate_passed,
        "quality_regressions": quality_regressions,
        "cold_baseline": cold_record,
        "comparison": {
            "status_accuracy": {"baseline": base_status_acc, "candidate": cand_status_acc},
            "generation_ms": {"baseline": base_gen_stats, "candidate": cand_gen_stats},
            "total_ms": {"baseline": base_tot_stats, "candidate": cand_tot_stats},
            "ttft_ms": {"candidate": cand_ttft_stats},
            "avg_eval_count": {
                "baseline": round(sum(r["eval_count"] for r in baseline_results) / len(baseline_results), 1),
                "candidate": round(sum(r["eval_count"] for r in candidate_results) / len(candidate_results), 1),
            },
            "avg_output_tokens": {
                "baseline": round(sum(r["eval_count"] for r in baseline_results) / len(baseline_results), 1),
                "candidate": round(sum(r["eval_count"] for r in candidate_results) / len(candidate_results), 1),
            },
            "avg_context_chars": {
                "baseline": round(sum(r["context_chars"] for r in baseline_results) / len(baseline_results), 1),
                "candidate": round(sum(r["context_chars"] for r in candidate_results) / len(candidate_results), 1),
            },
            "avg_answer_chars": {
                "baseline": round(sum(r["answer_chars"] for r in baseline_results) / len(baseline_results), 1),
                "candidate": round(sum(r["answer_chars"] for r in candidate_results) / len(candidate_results), 1),
            },
        },
    }

    # Save outputs
    with open(bench_file, "w", encoding="utf-8") as f:
        json.dump({
            "cold_baseline": cold_record,
            "baseline_results": baseline_results,
            "candidate_results": candidate_results,
        }, f, ensure_ascii=False, indent=2)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    # Export CSV
    fieldnames = [
        "id", "query", "category",
        "baseline_status", "candidate_status",
        "baseline_gen_ms", "candidate_gen_ms", "candidate_ttft_ms",
        "baseline_total_ms", "candidate_total_ms",
        "baseline_context_chars", "candidate_context_chars",
        "baseline_eval_count", "candidate_eval_count",
        "baseline_support_complete", "candidate_support_complete",
    ]
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b, c in zip(baseline_results, candidate_results):
            writer.writerow({
                "id": b["id"],
                "query": b["query"],
                "category": b["category"],
                "baseline_status": b["actual_status"],
                "candidate_status": c["actual_status"],
                "baseline_gen_ms": b["generation_ms"],
                "candidate_gen_ms": c["generation_ms"],
                "candidate_ttft_ms": c["ttft_ms"],
                "baseline_total_ms": b["total_ms"],
                "candidate_total_ms": c["total_ms"],
                "baseline_context_chars": b["context_chars"],
                "candidate_context_chars": c["context_chars"],
                "baseline_eval_count": b["eval_count"],
                "candidate_eval_count": c["eval_count"],
                "baseline_support_complete": b["retrieval_support_complete"],
                "candidate_support_complete": c["retrieval_support_complete"],
            })

    # Print summary table
    print_separator("PRODUCTION TUNING BENCHMARK REPORT")
    print(f"{'Metric':<35} | {'Warm Baseline':<20} | {'Warm Candidate':<20}")
    print("-" * 80)
    print(f"{'Status Accuracy':<35} | {base_status_acc * 100:.1f}%{'':<15} | {cand_status_acc * 100:.1f}%")
    print(f"{'TTFT Avg / P50 / P95 (ms)':<35} | {'N/A':<20} | {cand_ttft_stats['avg']:.0f} / {cand_ttft_stats['p50']:.0f} / {cand_ttft_stats['p95']:.0f}")
    print(f"{'Gen Latency Avg / P50 (ms)':<35} | {base_gen_stats['avg']:.0f} / {base_gen_stats['p50']:.0f}{'':<10} | {cand_gen_stats['avg']:.0f} / {cand_gen_stats['p50']:.0f}")
    print(f"{'Total Latency Avg / P50 (ms)':<35} | {base_tot_stats['avg']:.0f} / {base_tot_stats['p50']:.0f}{'':<10} | {cand_tot_stats['avg']:.0f} / {cand_tot_stats['p50']:.0f}")
    print(f"{'Avg Context Chars':<35} | {summary_data['comparison']['avg_context_chars']['baseline']:<20} | {summary_data['comparison']['avg_context_chars']['candidate']}")
    print(f"{'Avg Output Tokens (eval_count)':<35} | {summary_data['comparison']['avg_output_tokens']['baseline']:<20} | {summary_data['comparison']['avg_output_tokens']['candidate']}")
    print("-" * 80)
    print(f"QUALITY GATE STATUS: {'PASSED' if quality_gate_passed else 'FAILED'}")
    if quality_regressions:
        for qr in quality_regressions:
            print(f"  - {qr}")
    print(f"\nBenchmark artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
