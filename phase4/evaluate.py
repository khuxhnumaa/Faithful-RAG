import sys, os, csv, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))

from rag_p4 import run_full_pipeline


# ── Evaluation queries ────────────────────────────────────────────────────────
# ADD YOUR OWN QUERIES HERE.
# "grounded"     = your uploaded document CAN answer this
# "hallucinated" = your document CANNOT answer this (out-of-scope)

EVAL_QUERIES = [
    # ── In-scope (grounded) — adjust to match your uploaded document ──────────
    {"query": "What is the main contribution of this paper?",         "label": "grounded"},
    {"query": "What method or approach does this paper propose?",     "label": "grounded"},
    {"query": "What experiments or datasets were used?",              "label": "grounded"},
    {"query": "What problem does this research aim to solve?",        "label": "grounded"},
    {"query": "What are the results or performance of this method?",  "label": "grounded"},
    {"query": "What are the limitations mentioned in this paper?",    "label": "grounded"},
    {"query": "Who are the authors of this paper?",                   "label": "grounded"},
    {"query": "What is the conclusion of this paper?",                "label": "grounded"},

    # ── Out-of-scope (hallucinated) — universally not in any research paper ───
    {"query": "What is the current stock price of Tesla?",            "label": "hallucinated"},
    {"query": "Who won the FIFA World Cup in 2022?",                  "label": "hallucinated"},
    {"query": "What is the recipe for biryani?",                      "label": "hallucinated"},
    {"query": "What is the weather forecast for tomorrow?",           "label": "hallucinated"},
    {"query": "Who is the Prime Minister of India right now?",        "label": "hallucinated"},
    {"query": "What is the boiling point of nitrogen?",               "label": "hallucinated"},
    {"query": "How do I install Python on Windows?",                  "label": "hallucinated"},
    {"query": "What movies won the Oscar in 2023?",                   "label": "hallucinated"},
]


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_evaluation(
    queries:          list,
    db_path:          str  = "../week1/vectordb",
    skip_nli_filter:  bool = False,
    output_csv:       str  = "eval_results.csv",
) -> dict:
    results = []
    total   = len(queries)

    print(f"\n{'='*70}")
    print(f"EVALUATION — {total} queries")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    for i, item in enumerate(queries, 1):
        query = item["query"]
        label = item["label"]   # ground truth

        print(f"\n[{i}/{total}] {label.upper()} | {query[:60]}...")

        try:
            result = run_full_pipeline(
                query=query,
                db_path=db_path,
                sim_threshold=0.60,
                skip_nli_filter=skip_nli_filter,
                verbose=False,   # quiet during evaluation
            )

            # Map verdict to binary prediction
            # grounded = not hallucinated
            # borderline = treat as grounded (we're not sure)
            # hallucinated = hallucinated
            predicted_label = (
                "hallucinated"
                if result.verdict == "hallucinated"
                else "grounded"
            )

            correct = (predicted_label == label)

            results.append({
                "query":          query,
                "label":          label,
                "predicted":      predicted_label,
                "verdict":        result.verdict,
                "faithfulness":   result.faithfulness,
                "combined":       result.combined_score,
                "correct":        correct,
                "requery":        result.result.requery_attempted,
            })

            status = "✓ CORRECT" if correct else "✗ WRONG"
            print(f"  Label={label} | Predicted={predicted_label} | "
                  f"Score={result.combined_score:.3f} | {status}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "query": query, "label": label,
                "predicted": "error", "verdict": "error",
                "faithfulness": 0, "combined": 0,
                "correct": False, "requery": False,
            })

    # ── Compute metrics ────────────────────────────────────────────────────────

    # True positives: predicted hallucinated, actually hallucinated
    TP = sum(1 for r in results
             if r["predicted"] == "hallucinated" and r["label"] == "hallucinated")
    # False positives: predicted hallucinated, actually grounded
    FP = sum(1 for r in results
             if r["predicted"] == "hallucinated" and r["label"] == "grounded")
    # False negatives: predicted grounded, actually hallucinated
    FN = sum(1 for r in results
             if r["predicted"] == "grounded" and r["label"] == "hallucinated")
    # True negatives: predicted grounded, actually grounded
    TN = sum(1 for r in results
             if r["predicted"] == "grounded" and r["label"] == "grounded")

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (TP + TN) / total if total > 0 else 0.0

    # ── Print results table ────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"{'Query':<42} {'Label':<12} {'Pred':<12} {'Score':<7} OK?")
    print("-"*70)

    for r in results:
        ok = "✓" if r["correct"] else "✗"
        print(
            f"{r['query'][:41]:<42} "
            f"{r['label']:<12} "
            f"{r['predicted']:<12} "
            f"{r['combined']:.3f}   "
            f"{ok}"
        )

    print(f"\n{'─'*70}")
    print(f"  Total queries:   {total}")
    print(f"  Correct:         {sum(1 for r in results if r['correct'])}")
    print(f"  Accuracy:        {accuracy*100:.1f}%")
    print(f"\n  True Positives:  {TP}  (hallucinated correctly caught)")
    print(f"  False Positives: {FP}  (grounded answer wrongly flagged)")
    print(f"  True Negatives:  {TN}  (grounded answer correctly passed)")
    print(f"  False Negatives: {FN}  (hallucinated answer wrongly passed)")
    print(f"\n  Precision:       {precision*100:.1f}%")
    print(f"  Recall:          {recall*100:.1f}%")
    print(f"  F1 Score:        {f1*100:.1f}%")
    print(f"{'='*70}\n")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {output_csv}")

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
        "results":   results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate hallucination detection system"
    )
    parser.add_argument("--skip_nli_filter", action="store_true",
                        help="Skip Week 2 NLI filter (faster)")
    parser.add_argument("--db_path",  default="../week1/vectordb")
    parser.add_argument("--output",   default="eval_results.csv",
                        help="Output CSV filename")
    args = parser.parse_args()

    print("\n Hallucination Detection Evaluation")
    print(" Make sure 'ollama serve' is running!\n")
    print(" TIP: Edit EVAL_QUERIES in this file to match your uploaded document.")
    print("      Label in-scope queries as 'grounded' and")
    print("      out-of-scope queries as 'hallucinated'.\n")

    run_evaluation(
        queries=EVAL_QUERIES,
        db_path=args.db_path,
        skip_nli_filter=args.skip_nli_filter,
        output_csv=args.output,
    )