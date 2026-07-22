

import sys, os, csv, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'week1'))

from rag_p4 import run_full_pipeline


# ══════════════════════════════════════════════════════════════════════════════
# ALL QUERIES — COMBINED FROM ALL PAPERS
# ══════════════════════════════════════════════════════════════════════════════

ALL_QUERIES = [

    # ── Paper 1: RAGTruth ─────────────────────────────────────────────────────
    {"query": "What is the main contribution of the RAGTruth paper?",      "label": "grounded", "paper": "RAGTruth"},
    {"query": "What datasets are used in the RAGTruth experiments?",       "label": "grounded", "paper": "RAGTruth"},
    {"query": "How is hallucination defined in RAGTruth?",                 "label": "grounded", "paper": "RAGTruth"},
    {"query": "What annotation strategy does RAGTruth propose?",           "label": "grounded", "paper": "RAGTruth"},
    {"query": "What are the limitations discussed in RAGTruth?",           "label": "grounded", "paper": "RAGTruth"},
    {"query": "What future work is suggested in the RAGTruth paper?",      "label": "grounded", "paper": "RAGTruth"},

    # ── Paper 2: Attention Is All You Need ────────────────────────────────────
    {"query": "What architecture is proposed in Attention Is All You Need?","label": "grounded", "paper": "Transformer"},
    {"query": "Why is self-attention used in the Transformer?",            "label": "grounded", "paper": "Transformer"},
    {"query": "What datasets were used to evaluate the Transformer model?","label": "grounded", "paper": "Transformer"},
    {"query": "What is positional encoding in the Transformer paper?",     "label": "grounded", "paper": "Transformer"},
    {"query": "How many encoder layers are used in the Transformer?",      "label": "grounded", "paper": "Transformer"},
    {"query": "What are the experimental results of the Transformer?",     "label": "grounded", "paper": "Transformer"},

    # ── Paper 3: BERT ─────────────────────────────────────────────────────────
    {"query": "What is masked language modeling in BERT?",                 "label": "grounded", "paper": "BERT"},
    {"query": "What is next sentence prediction in BERT?",                 "label": "grounded", "paper": "BERT"},
    {"query": "What datasets were used during BERT pretraining?",          "label": "grounded", "paper": "BERT"},
    {"query": "How does BERT differ from previous language models?",       "label": "grounded", "paper": "BERT"},
    {"query": "What benchmark tasks were evaluated in the BERT paper?",    "label": "grounded", "paper": "BERT"},
    {"query": "What are the key findings of the BERT paper?",              "label": "grounded", "paper": "BERT"},

    # ── Paper 4: ResNet ───────────────────────────────────────────────────────
    {"query": "What problem does residual learning solve in ResNet?",      "label": "grounded", "paper": "ResNet"},
    {"query": "What is a residual block in ResNet?",                       "label": "grounded", "paper": "ResNet"},
    {"query": "Which datasets were used to evaluate ResNet?",              "label": "grounded", "paper": "ResNet"},
    {"query": "How deep is the largest ResNet network?",                   "label": "grounded", "paper": "ResNet"},
    {"query": "What improvements were reported in the ResNet paper?",      "label": "grounded", "paper": "ResNet"},
    {"query": "What are the conclusions of the ResNet paper?",             "label": "grounded", "paper": "ResNet"},

    # ── Paper 5: YOLOv1 ──────────────────────────────────────────────────────
    {"query": "What object detection framework does YOLO propose?",        "label": "grounded", "paper": "YOLOv1"},
    {"query": "How does YOLO differ from previous object detectors?",      "label": "grounded", "paper": "YOLOv1"},
    {"query": "Which datasets were used to evaluate YOLO?",                "label": "grounded", "paper": "YOLOv1"},
    {"query": "What mAP does YOLO report in the paper?",                   "label": "grounded", "paper": "YOLOv1"},
    {"query": "What are the strengths of the YOLO detection system?",      "label": "grounded", "paper": "YOLOv1"},
    {"query": "What weaknesses does the YOLO paper discuss?",              "label": "grounded", "paper": "YOLOv1"},

    # ── Paper 6: AlexNet ─────────────────────────────────────────────────────
    {"query": "What challenge did AlexNet win?",                           "label": "grounded", "paper": "AlexNet"},
    {"query": "Which activation function was introduced in AlexNet?",      "label": "grounded", "paper": "AlexNet"},
    {"query": "Which dataset was used to train AlexNet?",                  "label": "grounded", "paper": "AlexNet"},
    {"query": "What regularization methods were used in AlexNet?",         "label": "grounded", "paper": "AlexNet"},
    {"query": "What hardware was used to train AlexNet?",                  "label": "grounded", "paper": "AlexNet"},
    {"query": "What performance did AlexNet achieve?",                     "label": "grounded", "paper": "AlexNet"},

    # ── Paper 7: EfficientNet ─────────────────────────────────────────────────
    {"query": "What scaling method does EfficientNet propose?",            "label": "grounded", "paper": "EfficientNet"},
    {"query": "What is compound scaling in EfficientNet?",                 "label": "grounded", "paper": "EfficientNet"},
    {"query": "Which benchmark datasets were used for EfficientNet?",      "label": "grounded", "paper": "EfficientNet"},
    {"query": "How does EfficientNet improve computational efficiency?",   "label": "grounded", "paper": "EfficientNet"},
    {"query": "What results did EfficientNet achieve?",                    "label": "grounded", "paper": "EfficientNet"},
    {"query": "What limitations does the EfficientNet paper discuss?",     "label": "grounded", "paper": "EfficientNet"},

    # ── Paper 8: Segment Anything (SAM) ──────────────────────────────────────
    {"query": "What problem does Segment Anything Model solve?",           "label": "grounded", "paper": "SAM"},
    {"query": "What is the SA-1B dataset in the SAM paper?",               "label": "grounded", "paper": "SAM"},
    {"query": "How does promptable segmentation work in SAM?",             "label": "grounded", "paper": "SAM"},
    {"query": "How many images are in the SAM dataset?",                   "label": "grounded", "paper": "SAM"},
    {"query": "What experiments were conducted in the SAM paper?",         "label": "grounded", "paper": "SAM"},
    {"query": "What are the conclusions of the SAM paper?",                "label": "grounded", "paper": "SAM"},

    # ── Paper 9: GPT-3 ───────────────────────────────────────────────────────
    {"query": "How many parameters does GPT-3 have?",                      "label": "grounded", "paper": "GPT-3"},
    {"query": "What is in-context learning in GPT-3?",                     "label": "grounded", "paper": "GPT-3"},
    {"query": "What benchmark datasets were used to evaluate GPT-3?",      "label": "grounded", "paper": "GPT-3"},
    {"query": "How does few-shot learning work in GPT-3?",                 "label": "grounded", "paper": "GPT-3"},
    {"query": "What are the limitations of GPT-3?",                        "label": "grounded", "paper": "GPT-3"},
    {"query": "What conclusions does the GPT-3 paper draw?",               "label": "grounded", "paper": "GPT-3"},

    # ── Paper 10: A Decade of Deep Learning (survey) ─────────────────────────
    {"query": "What is the main theme of this survey paper?",              "label": "grounded", "paper": "Decade of DL"},
    {"query": "What deep learning milestones are covered?",                "label": "grounded", "paper": "Decade of DL"},
    {"query": "What applications of deep learning are discussed?",         "label": "grounded", "paper": "Decade of DL"},
    {"query": "What challenges in deep learning are mentioned?",           "label": "grounded", "paper": "Decade of DL"},
    {"query": "What future directions are discussed in the paper?",        "label": "grounded", "paper": "Decade of DL"},
    {"query": "What conclusions are drawn in this survey?",                "label": "grounded", "paper": "Decade of DL"},

    # ══════════════════════════════════════════════════════════════════════════
    # HALLUCINATION QUERIES — universally unrelated to any research paper
    # ══════════════════════════════════════════════════════════════════════════
    {"query": "Who won the FIFA World Cup in 2022?",                       "label": "hallucinated", "paper": "NONE"},
    {"query": "What is the weather forecast for tomorrow in Delhi?",       "label": "hallucinated", "paper": "NONE"},
    {"query": "What is the recipe for chicken biryani?",                   "label": "hallucinated", "paper": "NONE"},
    {"query": "Who is the Prime Minister of India right now?",             "label": "hallucinated", "paper": "NONE"},
    {"query": "What is the current stock price of Tesla?",                 "label": "hallucinated", "paper": "NONE"},
    {"query": "How do I install Ubuntu 22 on a laptop?",                   "label": "hallucinated", "paper": "NONE"},
    {"query": "What movies won the Oscar for Best Picture in 2023?",       "label": "hallucinated", "paper": "NONE"},
    {"query": "What is the boiling point of nitrogen?",                    "label": "hallucinated", "paper": "NONE"},
    {"query": "Who is the CEO of OpenAI right now?",                       "label": "hallucinated", "paper": "NONE"},
    {"query": "What is the population of Mumbai in 2024?",                 "label": "hallucinated", "paper": "NONE"},
]

# ── Benchmark comparison values (from RAGTruth paper, Table 4) ───────────────
RAGTRUTH_BENCHMARKS = {
    "name":        "RAGTruth best detector (Wu et al., 2024)",
    "precision":   0.851,
    "recall":      0.724,
    "f1":          0.782,
    "accuracy":    0.796,
    "note":        "arXiv:2401.00396 — Table 4, best performing model on their benchmark corpus"
}


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_unified_evaluation(
    db_path:         str,
    sim_threshold:   float = 0.50,
    skip_nli_filter: bool  = True,
    output_csv:      str   = "unified_eval_results.csv",
):
    total   = len(ALL_QUERIES)
    grounded_count     = sum(1 for q in ALL_QUERIES if q["label"] == "grounded")
    hallucinated_count = sum(1 for q in ALL_QUERIES if q["label"] == "hallucinated")

    print(f"\n{'='*75}")
    print(f"  UNIFIED EVALUATION — ALL PAPERS")
    print(f"{'='*75}")
    print(f"  Total queries:      {total}")
    print(f"  Grounded:           {grounded_count}  (from 10 papers, 6 per paper)")
    print(f"  Hallucinated:       {hallucinated_count}  (universally out-of-scope)")
    print(f"  DB path:            {db_path}")
    print(f"  Sim threshold:      {sim_threshold}")
    print(f"  NLI filter:         {'OFF' if skip_nli_filter else 'ON'}")
    print(f"  Started:            {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*75}\n")

    results = []

    for i, item in enumerate(ALL_QUERIES, 1):
        query = item["query"]
        label = item["label"]
        paper = item["paper"]

        print(f"[{i:03d}/{total}] [{label.upper():<12}] [{paper:<12}] {query[:50]}...")

        try:
            result = run_full_pipeline(
                query           = query,
                db_path         = db_path,
                sim_threshold   = sim_threshold,
                skip_nli_filter = skip_nli_filter,
                verbose         = False,
            )
            predicted = "hallucinated" if result.verdict == "hallucinated" else "grounded"
            correct   = (predicted == label)

            results.append({
                "paper":        paper,
                "query":        query,
                "label":        label,
                "predicted":    predicted,
                "verdict":      result.verdict,
                "faithfulness": round(result.faithfulness, 4),
                "combined":     round(result.combined_score, 4),
                "correct":      correct,
            })
            status = "✓" if correct else "✗"
            print(f"           → {predicted:<12} score={result.combined_score:.3f}  {status}")

        except Exception as e:
            print(f"           → ERROR: {e}")
            results.append({
                "paper": paper, "query": query, "label": label,
                "predicted": "error", "verdict": "error",
                "faithfulness": 0.0, "combined": 0.0, "correct": False,
            })

    # ── Compute metrics ────────────────────────────────────────────────────────
    TP = sum(1 for r in results if r["predicted"]=="hallucinated" and r["label"]=="hallucinated")
    FP = sum(1 for r in results if r["predicted"]=="hallucinated" and r["label"]=="grounded")
    FN = sum(1 for r in results if r["predicted"]=="grounded"     and r["label"]=="hallucinated")
    TN = sum(1 for r in results if r["predicted"]=="grounded"     and r["label"]=="grounded")

    precision = TP/(TP+FP) if (TP+FP)>0 else 0.0
    recall    = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    accuracy  = (TP+TN)/total if total>0 else 0.0
    correct_count = sum(1 for r in results if r["correct"])

    # ── Per-paper breakdown ────────────────────────────────────────────────────
    papers_seen = []
    for p in [q["paper"] for q in ALL_QUERIES]:
        if p not in papers_seen:
            papers_seen.append(p)

    print(f"\n\n{'='*75}")
    print(f"PER-PAPER BREAKDOWN (grounded queries only)")
    print(f"{'='*75}")
    print(f"{'Paper':<18} {'Total':<7} {'Correct':<9} {'Accuracy':<10} {'Avg Faith'}")
    print("-"*60)

    for paper in papers_seen:
        if paper == "NONE":
            continue
        paper_results = [r for r in results if r["paper"] == paper and r["label"] == "grounded"]
        if not paper_results:
            continue
        p_correct   = sum(1 for r in paper_results if r["correct"])
        p_accuracy  = p_correct / len(paper_results) if paper_results else 0
        p_faith_avg = sum(r["faithfulness"] for r in paper_results) / len(paper_results)
        print(f"{paper:<18} {len(paper_results):<7} {p_correct:<9} {p_accuracy*100:<10.1f}% {p_faith_avg:.3f}")

    # ── Hallucination queries breakdown ───────────────────────────────────────
    hall_results = [r for r in results if r["label"] == "hallucinated"]
    hall_correct = sum(1 for r in hall_results if r["correct"])
    print("-"*60)
    print(f"{'OUT-OF-SCOPE':<18} {len(hall_results):<7} {hall_correct:<9} {hall_correct/len(hall_results)*100:.1f}%      —")

    # ── Overall results ────────────────────────────────────────────────────────
    print(f"\n\n{'='*75}")
    print(f"OVERALL RESULTS — {total} QUERIES")
    print(f"{'='*75}")
    print(f"  Total queries:       {total}")
    print(f"  Correct predictions: {correct_count}")
    print(f"  Accuracy:            {accuracy*100:.1f}%")
    print(f"\n  True Positives (TP): {TP}   hallucinated correctly caught")
    print(f"  False Positives (FP):{FP}   grounded answer wrongly flagged")
    print(f"  True Negatives (TN): {TN}   grounded answer correctly passed")
    print(f"  False Negatives (FN):{FN}   hallucinated answer wrongly passed")
    print(f"\n  Precision:           {precision*100:.1f}%")
    print(f"  Recall:              {recall*100:.1f}%")
    print(f"  F1 Score:            {f1*100:.1f}%")

    # ── Benchmark comparison ───────────────────────────────────────────────────
    bm = RAGTRUTH_BENCHMARKS
    print(f"\n\n{'='*75}")
    print(f"BENCHMARK COMPARISON — {bm['name']}")
    print(f"{'='*75}")
    print(f"  {bm['note']}")
    print()
    print(f"  {'Metric':<12} {'Our System':<16} {'RAGTruth Benchmark':<20} {'Difference'}")
    print(f"  {'─'*60}")

    def diff_str(ours, theirs):
        d = (ours - theirs) * 100
        return f"+{d:.1f}%" if d >= 0 else f"{d:.1f}%"

    print(f"  {'Precision':<12} {precision*100:<16.1f}% {bm['precision']*100:<20.1f}% {diff_str(precision, bm['precision'])}")
    print(f"  {'Recall':<12} {recall*100:<16.1f}% {bm['recall']*100:<20.1f}% {diff_str(recall, bm['recall'])}")
    print(f"  {'F1 Score':<12} {f1*100:<16.1f}% {bm['f1']*100:<20.1f}% {diff_str(f1, bm['f1'])}")
    print(f"  {'Accuracy':<12} {accuracy*100:<16.1f}% {bm['accuracy']*100:<20.1f}% {diff_str(accuracy, bm['accuracy'])}")
    print(f"\n  Note: RAGTruth benchmarks are on their own annotated dataset.")
    print(f"  Our evaluation is on a broader cross-domain benchmark (10 papers).")
    print(f"  Direct comparison is approximate — task setups differ slightly.")
    print(f"{'='*75}\n")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Detailed results saved: {output_csv}")

    # ── Save summary ───────────────────────────────────────────────────────────
    summary_csv = output_csv.replace(".csv", "_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Our System", "RAGTruth Benchmark", "Difference"])
        writer.writerow(["Precision",  f"{precision*100:.1f}%",  f"{bm['precision']*100:.1f}%",  diff_str(precision, bm['precision'])])
        writer.writerow(["Recall",     f"{recall*100:.1f}%",     f"{bm['recall']*100:.1f}%",     diff_str(recall, bm['recall'])])
        writer.writerow(["F1 Score",   f"{f1*100:.1f}%",         f"{bm['f1']*100:.1f}%",         diff_str(f1, bm['f1'])])
        writer.writerow(["Accuracy",   f"{accuracy*100:.1f}%",   f"{bm['accuracy']*100:.1f}%",   diff_str(accuracy, bm['accuracy'])])
        writer.writerow(["TP", TP, "—", "—"])
        writer.writerow(["FP", FP, "—", "—"])
        writer.writerow(["FN", FN, "—", "—"])
        writer.writerow(["TN", TN, "—", "—"])
    print(f"Summary saved:          {summary_csv}")

    return {
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "total": total, "correct": correct_count,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified multi-paper hallucination detection evaluation"
    )
    parser.add_argument(
        "--db_path",
        default="/Users/khushnumaparveen/Desktop/Internship/Hallucination_RAG/week1/vectordb",
        help="Path to FAISS vector DB folder containing ALL papers"
    )
    parser.add_argument("--threshold", default=0.30, type=float)
    parser.add_argument("--output",    default="unified_eval_results.csv")
    args = parser.parse_args()

    print(f"\n RAG Hallucination Detection — Unified Evaluation")
    print(f" Make sure 'ollama serve' is running!\n")

    # Quick check
    if not os.path.exists(os.path.join(args.db_path, "index.faiss")):
        print(f"[ERROR] No FAISS index found at: {args.db_path}")
        print(f"Run this first:")
        print(f"  cd week1")
        print(f"  python ingest.py --docs_dir ./docs_multi --db_path ./vectordb")
        exit(1)

    run_unified_evaluation(
        db_path         = args.db_path,
        sim_threshold   = args.threshold,
        skip_nli_filter = True,
        output_csv      = args.output,
    )