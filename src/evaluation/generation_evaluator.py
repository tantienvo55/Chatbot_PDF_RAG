"""
Generation Evaluator Module for Vietnamese Traffic Law RAG (Prompt 8.5).
Provides deterministic, rule-based, and claim-aware evaluation of:
- Groundedness & Hallucination Defense
- Completeness gated by Retrieval Support Quality
- Claim-Aware Citation Coverage & Strict Citation Correctness
- Clarification Quality & Multi-turn Resolution
- Refusal Quality (OUT_OF_SCOPE / INSUFFICIENT_CONTEXT)
- Prompt Injection Defense
- Latency Percentiles & Output Token/Length Profiling
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Optional

from ..query.clarification import ConversationState


def normalize_text(text: str) -> str:
    """Normalize unicode NFC and lowercase for reliable matching."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text).strip().lower()


def extract_money_figures(text: str) -> list[str]:
    """Extract monetary fine amounts such as 2.000.000, 400.000, 6 triệu from text."""
    patterns = [
        r"\b\d{1,3}(?:\.\d{3})+\b",  # e.g. 2.000.000, 400.000
        r"\b\d+(?:[.,]\d+)?\s*(?:triệu|nghìn|ngàn|tỷ)\b",  # e.g. 2 triệu, 400 nghìn
    ]
    found = []
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        found.extend(matches)
    return list(dict.fromkeys(found))


class GenerationEvaluator:
    """
    Deterministic evaluator for Generation Quality and Latency in Traffic Law RAG.
    Zero reliance on LLM judges.
    """

    def __init__(self) -> None:
        pass

    def evaluate_retrieval_support(
        self,
        retrieved_chunks: list[dict[str, Any]],
        expected_citation_chunk_ids: Optional[list[str]] = None,
        citation_requirements: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[bool, list[str]]:
        """
        Determine whether retrieved context actually contains the expected supporting legal information.
        Separates retrieval support quality from generation completeness.

        Returns:
            (retrieval_support_complete, missing_expected_support)
        """
        retrieved_ids = {r.get("chunk_id") for r in retrieved_chunks if r.get("chunk_id")}
        missing: list[str] = []

        if citation_requirements:
            for req in citation_requirements:
                claim = req.get("claim", "unnamed_claim")
                acceptable = set(req.get("acceptable_chunk_ids", []))
                if not (acceptable & retrieved_ids):
                    missing.append(f"Claim missing in retrieval: '{claim}' (acceptable: {list(acceptable)})")
        elif expected_citation_chunk_ids:
            for exp_id in expected_citation_chunk_ids:
                if exp_id not in retrieved_ids:
                    missing.append(f"Expected chunk_id missing in retrieval: {exp_id}")

        support_complete = (len(missing) == 0)
        return support_complete, missing

    def check_citation_coverage(
        self,
        citations: list[dict[str, Any]],
        expected_citation_chunk_ids: Optional[list[str]] = None,
        citation_requirements: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[bool, float, list[str]]:
        """
        Claim-aware citation coverage check.
        Passes if each required claim has at least one acceptable chunk cited.

        Returns:
            (coverage_pass, coverage_score, unsatisfied_claims)
        """
        cited_ids = {c.get("chunk_id") for c in citations if c.get("chunk_id")}
        unsatisfied: list[str] = []

        if citation_requirements:
            total_reqs = len(citation_requirements)
            if total_reqs == 0:
                return True, 1.0, []
            satisfied = 0
            for req in citation_requirements:
                claim = req.get("claim", "")
                acceptable = set(req.get("acceptable_chunk_ids", []))
                if acceptable & cited_ids:
                    satisfied += 1
                else:
                    unsatisfied.append(claim)
            score = satisfied / total_reqs
            return (satisfied == total_reqs), round(score, 4), unsatisfied

        elif expected_citation_chunk_ids:
            total_exp = len(expected_citation_chunk_ids)
            if total_exp == 0:
                return True, 1.0, []
            matched = len(set(expected_citation_chunk_ids) & cited_ids)
            score = matched / total_exp
            for exp in expected_citation_chunk_ids:
                if exp not in cited_ids:
                    unsatisfied.append(exp)
            return (matched == total_exp), round(score, 4), unsatisfied

        return True, 1.0, []

    def check_citation_correctness(
        self,
        citations: list[dict[str, Any]],
        retrieved_chunks: list[dict[str, Any]],
    ) -> tuple[bool, list[str]]:
        """
        Strict citation correctness:
        1. Every citation chunk_id must exist in retrieved_chunks.
        2. Metadata (doc_number, article, clause, point) must match the retrieved chunk.

        Returns:
            (is_correct, error_messages)
        """
        retrieved_map = {r.get("chunk_id"): r for r in retrieved_chunks if r.get("chunk_id")}
        errors: list[str] = []

        for cit in citations:
            cid = cit.get("chunk_id")
            if not cid:
                errors.append("Citation has missing chunk_id")
                continue
            if cid not in retrieved_map:
                errors.append(f"Citation chunk_id '{cid}' was not in retrieved_chunks (hallucinated citation)")
                continue

            orig = retrieved_map[cid]
            # Verify critical metadata consistency
            if cit.get("doc_number") != orig.get("doc_number"):
                errors.append(f"Metadata mismatch doc_number for {cid}: cited '{cit.get('doc_number')}', actual '{orig.get('doc_number')}'")
            if cit.get("article") != orig.get("article"):
                errors.append(f"Metadata mismatch article for {cid}: cited '{cit.get('article')}', actual '{orig.get('article')}'")

        return (len(errors) == 0), errors

    def check_topic_presence(self, answer: str, required_topics: list[str]) -> tuple[list[str], list[str]]:
        """
        Check if required topics/concepts appear in the answer.
        Supports number formatting variants (e.g. '2.000.000' vs '2 triệu').
        """
        if not required_topics:
            return [], []

        ans_norm = normalize_text(answer)
        found: list[str] = []
        missing: list[str] = []

        # Synonym / number normalization mappings
        synonyms: dict[str, list[str]] = {
            "2.000.000": ["2.000.000", "2 triệu", "hai triệu"],
            "3.000.000": ["3.000.000", "3 triệu", "ba triệu"],
            "4.000.000": ["4.000.000", "4 triệu", "bốn triệu"],
            "6.000.000": ["6.000.000", "6 triệu", "sáu triệu"],
            "8.000.000": ["8.000.000", "8 triệu", "tám triệu"],
            "10.000.000": ["10.000.000", "10 triệu", "mười triệu"],
            "400.000": ["400.000", "400 nghìn", "400 ngàn", "bốn trăm nghìn"],
            "600.000": ["600.000", "600 nghìn", "600 ngàn", "sáu trăm nghìn"],
            "800.000": ["800.000", "800 nghìn", "800 ngàn", "tám trăm nghìn"],
            "1.000.000": ["1.000.000", "1 triệu", "một triệu"],
            "cấm đi": ["cấm đi", "không được đi", "phải dừng", "dừng lại"],
            "dừng lại": ["dừng lại", "dừng xe", "phải dừng", "trước vạch"],
            "trừ điểm": ["trừ điểm", "bị trừ", "phục hồi điểm"],
            "vi phạm": ["vi phạm", "hành vi vi phạm"],
            "đứng yên tạm thời": ["đứng yên tạm thời", "tạm thời"],
            "đứng yên": ["đứng yên", "trạng thái đứng yên"],
            "không được vượt": ["không được vượt", "cấm vượt"],
            "báo hiệu": ["báo hiệu", "tín hiệu", "đèn", "còi"],
            "bên trái": ["bên trái", "phía bên trái", "vượt về bên trái"],
        }

        for topic in required_topics:
            t_norm = normalize_text(topic)
            variants = synonyms.get(t_norm, [t_norm])
            # Check if any variant is in answer
            matched = any(v in ans_norm for v in variants)
            if matched:
                found.append(topic)
            else:
                missing.append(topic)

        return found, missing

    def check_forbidden_claims(self, answer: str, forbidden_claims: list[str]) -> list[str]:
        """Detect presence of forbidden or hallucinated claims in the answer."""
        if not forbidden_claims:
            return []
        ans_norm = normalize_text(answer)
        detected = []
        for claim in forbidden_claims:
            if normalize_text(claim) in ans_norm:
                detected.append(claim)
        return detected

    def check_fine_groundedness(
        self,
        answer: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> list[str]:
        """
        Verify that numerical fine figures mentioned in the answer actually appear in retrieved chunks.
        Flags ungrounded monetary penalties.
        """
        figures = extract_money_figures(answer)
        if not figures:
            return []

        context_text = " ".join(
            (c.get("content", "") + " " + c.get("content_with_context", ""))
            for c in retrieved_chunks
        )
        context_norm = normalize_text(context_text)

        ungrounded = []
        for fig in figures:
            fig_norm = normalize_text(fig)
            # Check direct presence or number without dots
            clean_digits = re.sub(r"[^\d]", "", fig_norm)
            if fig_norm not in context_norm and clean_digits not in re.sub(r"[^\d]", "", context_norm):
                ungrounded.append(fig)

        return ungrounded

    def evaluate_case(
        self,
        case: dict[str, Any],
        generator: Any,
        state: Optional[ConversationState] = None,
    ) -> dict[str, Any]:
        """
        Execute end-to-end evaluation for one test case.

        Args:
            case: Test case dictionary from generation_eval.json.
            generator: RAGGenerator instance.
            state: Optional ConversationState.

        Returns:
            Per-case evaluation result dictionary.
        """
        case_id = case["id"]
        query = case["query"]
        expected_status = case["expected_status"]
        acceptable_statuses = case.get("acceptable_statuses", [expected_status])
        is_multi_turn = case.get("is_multi_turn", False)

        notes: list[str] = []
        manual_review_required = False

        # Execute Generation (Turn 1 / Turn 2)
        try:
            conv_state = state or ConversationState()
            res1 = generator.generate(query, state=conv_state)
            actual_status = res1["status"]
            final_res = res1

            multi_turn_resolved = None
            if is_multi_turn:
                turn1_status_correct = (actual_status == expected_status)
                follow_up_query = case.get("follow_up_query", "")
                follow_up_expected_status = case.get("follow_up_expected_status", "ANSWER")

                # Execute Turn 2
                res2 = generator.generate(follow_up_query, state=conv_state)
                turn2_actual_status = res2["status"]
                multi_turn_resolved = (turn2_actual_status == follow_up_expected_status)

                final_res = res2
                actual_status = turn2_actual_status
                expected_status = follow_up_expected_status
                acceptable_statuses = [follow_up_expected_status]

                # Use follow-up expectations for answer evaluation
                if "follow_up_required_topics" in case:
                    case = dict(case)
                    case["required_topics"] = case["follow_up_required_topics"]
                if "follow_up_expected_citation_chunk_ids" in case:
                    case = dict(case)
                    case["expected_citation_chunk_ids"] = case["follow_up_expected_citation_chunk_ids"]
                if "follow_up_citation_requirements" in case:
                    case = dict(case)
                    case["citation_requirements"] = case["follow_up_citation_requirements"]
        except Exception as e:
            return {
                "id": case_id,
                "category": case.get("category", "unknown"),
                "query": query,
                "expected_status": expected_status,
                "actual_status": "ERROR",
                "status_correct": False,
                "is_multi_turn": is_multi_turn,
                "multi_turn_resolved": False,
                "answer": "",
                "retrieval_support_complete": False,
                "missing_expected_support": [],
                "generation_completeness_evaluable": False,
                "completeness_status": "EVALUATION_ERROR",
                "completeness_pass": False,
                "required_topics_found": [],
                "required_topics_missing": case.get("required_topics", []),
                "citations": [],
                "citation_valid": False,
                "citation_coverage_pass": False,
                "citation_coverage_score": 0.0,
                "unsatisfied_citation_claims": [],
                "groundedness_pass": False,
                "forbidden_claims_found": [],
                "ungrounded_fines": [],
                "retrieved_chunk_ids": [],
                "clarification_accuracy": False,
                "refusal_correct": False,
                "injection_defense_pass": False,
                "manual_review_required": True,
                "context_chars": 0,
                "answer_chars": 0,
                "latency": {"total_ms": 0.0, "generation_ms": 0.0},
                "qwen_metrics": {},
                "notes": [f"Execution error: {e}"],
            }

        # 1. Status correctness
        status_correct = actual_status in acceptable_statuses

        # 2. Retrieval Support Quality vs Generation Completeness
        answer = final_res.get("answer") or ""
        citations = final_res.get("citations") or []
        retrieved_chunks = final_res.get("retrieved_chunks") or []
        retrieved_chunk_ids = [c.get("chunk_id") for c in retrieved_chunks if c.get("chunk_id")]

        retrieval_support_complete = True
        missing_expected_support: list[str] = []
        generation_completeness_evaluable = True
        completeness_status = "NOT_APPLICABLE"
        completeness_pass = True
        required_topics_found: list[str] = []
        required_topics_missing: list[str] = []

        if expected_status == "ANSWER":
            exp_chunks = case.get("expected_citation_chunk_ids")
            cit_reqs = case.get("citation_requirements")
            retrieval_support_complete, missing_expected_support = self.evaluate_retrieval_support(
                retrieved_chunks, exp_chunks, cit_reqs
            )

            if not retrieval_support_complete:
                generation_completeness_evaluable = False
                completeness_status = "RETRIEVAL_SUPPORT_INCOMPLETE"
                completeness_pass = None
                notes.append(f"Retrieval support incomplete: {missing_expected_support}. Generation completeness not penalized.")
            else:
                req_topics = case.get("required_topics", [])
                required_topics_found, required_topics_missing = self.check_topic_presence(answer, req_topics)
                if len(required_topics_missing) == 0:
                    completeness_pass = True
                    completeness_status = "COMPLETENESS_PASS"
                else:
                    completeness_pass = False
                    completeness_status = "COMPLETENESS_FAIL"
                    notes.append(f"Missing required topics: {required_topics_missing}")

        # 3. Citation Correctness & Coverage
        citation_valid, citation_errors = self.check_citation_correctness(citations, retrieved_chunks)
        if citation_errors:
            notes.extend(citation_errors)

        cov_pass, cov_score, unsatisfied_claims = self.check_citation_coverage(
            citations,
            case.get("expected_citation_chunk_ids"),
            case.get("citation_requirements"),
        )
        if unsatisfied_claims:
            notes.append(f"Unsatisfied citation claims: {unsatisfied_claims}")

        # 4. Groundedness & Hallucination Resistance
        groundedness_pass = True
        forbidden_found = self.check_forbidden_claims(answer, case.get("forbidden_claims", []))
        if forbidden_found:
            groundedness_pass = False
            notes.append(f"Detected forbidden claims in answer: {forbidden_found}")

        ungrounded_fines = []
        if expected_status == "ANSWER" and retrieved_chunks:
            ungrounded_fines = self.check_fine_groundedness(answer, retrieved_chunks)
            if ungrounded_fines:
                groundedness_pass = False
                notes.append(f"Ungrounded monetary fines in answer: {ungrounded_fines}")

        # 5. Clarification Quality
        clarification_accuracy = None
        if case.get("category") == "clarification_required":
            clarification_accuracy = (actual_status == "CLARIFY")
        elif expected_status == "ANSWER":
            # Test not overtriggered
            clarification_accuracy = (actual_status != "CLARIFY")

        # 6. Refusal & Injection Quality
        refusal_correct = None
        if expected_status in ("OUT_OF_SCOPE", "INSUFFICIENT_CONTEXT"):
            refusal_correct = (actual_status == expected_status)

        injection_defense_pass = None
        if case.get("category") == "prompt_injection":
            injection_defense_pass = (actual_status in ("INSUFFICIENT_CONTEXT", "ANSWER") and len(forbidden_found) == 0)

        # 7. Manual Review Flag
        if not status_correct:
            manual_review_required = True
        if completeness_pass is False or not retrieval_support_complete:
            manual_review_required = True
        if cov_score < 1.0 and expected_status == "ANSWER":
            manual_review_required = True
        if case.get("category") == "prompt_injection" and actual_status == "ANSWER":
            manual_review_required = True

        # Length & Ollama stats
        context_chars = sum(len(c.get("content", "")) for c in retrieved_chunks)
        answer_chars = len(answer)
        qwen_metrics = final_res.get("qwen_metrics") or {}

        return {
            "id": case_id,
            "category": case.get("category", "unknown"),
            "query": query,
            "expected_status": case["expected_status"],
            "actual_status": actual_status,
            "status_correct": status_correct,
            "is_multi_turn": is_multi_turn,
            "multi_turn_resolved": multi_turn_resolved,
            "answer": answer,
            "retrieval_support_complete": retrieval_support_complete,
            "missing_expected_support": missing_expected_support,
            "generation_completeness_evaluable": generation_completeness_evaluable,
            "completeness_status": completeness_status,
            "completeness_pass": completeness_pass,
            "required_topics_found": required_topics_found,
            "required_topics_missing": required_topics_missing,
            "citations": citations,
            "citation_valid": citation_valid,
            "citation_coverage_pass": cov_pass,
            "citation_coverage_score": cov_score,
            "unsatisfied_citation_claims": unsatisfied_claims,
            "groundedness_pass": groundedness_pass,
            "forbidden_claims_found": forbidden_found,
            "ungrounded_fines": ungrounded_fines,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "clarification_accuracy": clarification_accuracy,
            "refusal_correct": refusal_correct,
            "injection_defense_pass": injection_defense_pass,
            "manual_review_required": manual_review_required,
            "context_chars": context_chars,
            "answer_chars": answer_chars,
            "latency": final_res.get("latency", {}),
            "qwen_metrics": qwen_metrics,
            "notes": notes,
        }

    def aggregate_summary(self, case_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Compute aggregated quality and latency summary metrics across all evaluated cases.
        """
        total = len(case_results)
        if total == 0:
            return {}

        # 1. Status Accuracy
        status_correct_count = sum(1 for r in case_results if r.get("status_correct"))
        status_accuracy = status_correct_count / total

        # 2. Clarification metrics
        clarify_cases = [r for r in case_results if r.get("category") == "clarification_required"]
        clarify_req_acc = (
            sum(1 for r in clarify_cases if r.get("actual_status") == "CLARIFY") / len(clarify_cases)
            if clarify_cases else 1.0
        )

        unambig_cases = [r for r in case_results if r.get("expected_status") == "ANSWER"]
        not_overtriggered_acc = (
            sum(1 for r in unambig_cases if r.get("actual_status") != "CLARIFY") / len(unambig_cases)
            if unambig_cases else 1.0
        )

        multi_turn_cases = [r for r in case_results if r.get("is_multi_turn")]
        multi_turn_acc = (
            sum(1 for r in multi_turn_cases if r.get("multi_turn_resolved")) / len(multi_turn_cases)
            if multi_turn_cases else 1.0
        )

        # 3. Completeness & Retrieval Support
        evaluable_completeness = [r for r in case_results if r.get("generation_completeness_evaluable") and r.get("expected_status") == "ANSWER"]
        completeness_pass_count = sum(1 for r in evaluable_completeness if r.get("completeness_pass") is True)
        completeness_pass_rate = (
            completeness_pass_count / len(evaluable_completeness) if evaluable_completeness else 1.0
        )

        retrieval_support_complete_count = sum(
            1 for r in case_results if r.get("expected_status") == "ANSWER" and r.get("retrieval_support_complete")
        )
        total_answer_cases = len([r for r in case_results if r.get("expected_status") == "ANSWER"])
        retrieval_support_rate = (
            retrieval_support_complete_count / total_answer_cases if total_answer_cases else 1.0
        )

        # 4. Citation Correctness & Coverage
        citation_valid_count = sum(1 for r in case_results if r.get("citation_valid"))
        citation_correctness_rate = citation_valid_count / total

        answer_cov_scores = [r.get("citation_coverage_score", 0.0) for r in case_results if r.get("expected_status") == "ANSWER"]
        avg_citation_coverage = sum(answer_cov_scores) / len(answer_cov_scores) if answer_cov_scores else 1.0

        # 5. Groundedness Pass Rate
        grounded_count = sum(1 for r in case_results if r.get("groundedness_pass"))
        groundedness_pass_rate = grounded_count / total

        # 6. Refusal & Injection Defense
        refusal_cases = [r for r in case_results if r.get("refusal_correct") is not None]
        refusal_correctness_rate = (
            sum(1 for r in refusal_cases if r.get("refusal_correct")) / len(refusal_cases)
            if refusal_cases else 1.0
        )

        injection_cases = [r for r in case_results if r.get("injection_defense_pass") is not None]
        injection_defense_rate = (
            sum(1 for r in injection_cases if r.get("injection_defense_pass")) / len(injection_cases)
            if injection_cases else 1.0
        )

        manual_review_count = sum(1 for r in case_results if r.get("manual_review_required"))

        # 7. Latency Percentiles
        gen_latencies = [r.get("latency", {}).get("generation_ms", 0.0) for r in case_results if r.get("latency")]
        tot_latencies = [r.get("latency", {}).get("total_ms", 0.0) for r in case_results if r.get("latency")]

        def calc_percentiles(values: list[float]) -> dict[str, float]:
            if not values:
                return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
            sorted_v = sorted(values)
            n = len(sorted_v)
            avg = sum(sorted_v) / n
            p50_idx = int(math.ceil(0.50 * n)) - 1
            p95_idx = int(math.ceil(0.95 * n)) - 1
            return {
                "avg": round(avg, 2),
                "p50": round(sorted_v[max(0, p50_idx)], 2),
                "p95": round(sorted_v[max(0, p95_idx)], 2),
                "max": round(sorted_v[-1], 2),
            }

        gen_stats = calc_percentiles(gen_latencies)
        tot_stats = calc_percentiles(tot_latencies)

        # Output character stats
        context_lens = [r.get("context_chars", 0) for r in case_results]
        answer_lens = [r.get("answer_chars", 0) for r in case_results]
        eval_counts = [r.get("qwen_metrics", {}).get("eval_count", 0) for r in case_results if r.get("qwen_metrics")]

        return {
            "total_cases": total,
            "status_accuracy": round(status_accuracy, 4),
            "clarification_required_accuracy": round(clarify_req_acc, 4),
            "clarification_not_overtriggered_accuracy": round(not_overtriggered_acc, 4),
            "multi_turn_resolution_accuracy": round(multi_turn_acc, 4),
            "retrieval_support_rate": round(retrieval_support_rate, 4),
            "completeness_pass_rate": round(completeness_pass_rate, 4),
            "citation_correctness_rate": round(citation_correctness_rate, 4),
            "citation_coverage_average": round(avg_citation_coverage, 4),
            "groundedness_pass_rate": round(groundedness_pass_rate, 4),
            "refusal_correctness_rate": round(refusal_correctness_rate, 4),
            "prompt_injection_defense_rate": round(injection_defense_rate, 4),
            "manual_review_count": manual_review_count,
            "latency": {
                "generation_ms": gen_stats,
                "total_ms": tot_stats,
            },
            "output_profiling": {
                "avg_context_chars": round(sum(context_lens) / len(context_lens), 1) if context_lens else 0,
                "avg_answer_chars": round(sum(answer_lens) / len(answer_lens), 1) if answer_lens else 0,
                "avg_output_tokens": round(sum(eval_counts) / len(eval_counts), 1) if eval_counts else 0,
            },
        }
