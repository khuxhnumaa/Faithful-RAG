"""
app.py — Complete RAG Hallucination Detection Dashboard
Week 4 + Week 5 combined into one file.

WHAT THIS FILE DOES:
    This is the final deliverable of the entire project.
    It is a Streamlit web app that brings together every component
    built across all four weeks into one interactive interface.

    Tab 1 — Query:
        User uploads any PDF or TXT document.
        System auto-ingests it into FAISS (Week 1: ingest.py).
        User types any question.
        Full pipeline runs: retrieve (W1) → sim filter (W2) →
        NLI filter (W2) → Llama generates (W1) → sentence split (W3) →
        NLI detection (W3) → RAGAS scores (W4) → verdict (W4).
        Results shown as: verdict badge, 4 score cards,
        colour-highlighted answer (green/orange/red per sentence),
        retrieved chunk detail panel, per-sentence NLI table.

    Tab 2 — Evaluate:
        User edits a list of labelled queries (grounded / hallucinated).
        System runs all queries through the full pipeline.
        Computes and displays: Precision, Recall, F1, Accuracy.
        Results downloadable as CSV for the project report.

    Tab 3 — History:
        Shows all queries asked in the current session with their
        verdicts, scores, and response time.

    Tab 4 — About:
        System description, paper citations, tech stack.

HOW TO RUN:
    pip install streamlit pypdf
    ollama serve          # in a separate terminal, keep running
    streamlit run app.py  # opens http://localhost:8501
"""

import streamlit as st
import sys, os, tempfile, time, csv, io
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / ".." / "week3"))
sys.path.insert(0, str(BASE / ".." / "week2"))
sys.path.insert(0, str(BASE / ".." / "week1"))

DB_PATH = str(BASE / ".." / "week1" / "vectordb_ui")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "RAG Hallucination Detector",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""<style>
.verdict-box { padding:14px 20px; border-radius:10px;
               margin-bottom:16px; font-size:18px; font-weight:600; }
.score-label { font-size:11px; color:#888; text-transform:uppercase;
               letter-spacing:.05em; margin-bottom:4px; }
.score-val   { font-size:28px; font-weight:600; margin-bottom:2px; }
.score-desc  { font-size:11px; color:#888; }
.sent-block  { line-height:2.3; font-size:15px; padding:12px;
               border-radius:8px; background:#fafafa; }
.chunk-box   { background:#f8f9fa; border-left:3px solid #ccc;
               padding:8px 12px; border-radius:4px; margin-bottom:8px; font-size:13px; }
.chunk-kept  { border-left-color:#1A7A3A; }
.chunk-drop  { border-left-color:#E24B4A; opacity:.6; }
.hist-item   { padding:10px 14px; border-radius:8px; border:.5px solid #e0e0e0;
               margin-bottom:8px; font-size:13px; }
</style>""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("doc_ready",False),("doc_name",""),("history",[]),("result",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — document upload + settings
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📄 Document")
    uploaded = st.file_uploader("Upload any PDF or TXT", type=["pdf","txt"])

    if uploaded:
        if uploaded.name != st.session_state.doc_name:
            tmp      = tempfile.mkdtemp()
            doc_path = os.path.join(tmp, uploaded.name)
            with open(doc_path,"wb") as f:
                f.write(uploaded.getvalue())
            with st.spinner(f"Indexing {uploaded.name} …"):
                try:
                    from ingest import load_documents, chunk_documents, build_vectorstore
                    docs   = load_documents(tmp)
                    chunks = chunk_documents(docs)
                    build_vectorstore(chunks, DB_PATH)
                    st.session_state.doc_ready = True
                    st.session_state.doc_name  = uploaded.name
                    st.session_state.history   = []
                    st.session_state.result    = None
                    st.success(f"✓ {len(chunks)} chunks indexed")
                except Exception as e:
                    st.error(f"Indexing error: {e}")
                    st.session_state.doc_ready = False
        else:
            st.success(f"✓ {st.session_state.doc_name}")
    elif os.path.exists(DB_PATH):
        st.session_state.doc_ready = True
        st.info(st.session_state.doc_name or "Previous document loaded")
    else:
        st.info("Upload a document to start")

    st.divider()
    st.markdown("## ⚙️ Settings")
    sim_threshold = st.slider("Similarity threshold", 0.40, 0.90, 0.60, 0.05,
        help="Chunks below this cosine score are dropped before generation.")
    skip_nli = st.checkbox("Skip NLI context filter (faster)", value=False,
        help="Disables Week 2 NLI filter — faster but may let noisy chunks through.")

    st.divider()
    st.markdown("## 📊 Verdict thresholds")
    st.markdown("| Score | Verdict |\n|---|---|\n| ≥ 0.75 | ✓ Grounded |\n| 0.50–0.75 | ⚠ Borderline |\n| < 0.50 | ✗ Hallucinated |")
    st.divider()
    st.caption("🔒 Fully local · No API key · No data sent externally")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🔍 RAG Hallucination Detector")
st.caption("Upload any document · Ask any question · See which sentences are grounded or hallucinated")

tab_q, tab_e, tab_h, tab_a = st.tabs(["🔎 Query","📊 Evaluate","🕘 History","ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUERY
# ══════════════════════════════════════════════════════════════════════════════
with tab_q:
    query = st.text_input(
        "Ask anything about your document",
        placeholder="e.g.  What is the main contribution of this paper?",
        disabled=not st.session_state.doc_ready,
    )
    c1, c2 = st.columns([1,6])
    with c1:
        run = st.button("Detect", type="primary",
                        disabled=not (st.session_state.doc_ready and bool(query.strip())),
                        use_container_width=True)
    with c2:
        if st.button("Clear"):
            st.session_state.result = None
            st.rerun()

    # ── Run pipeline ───────────────────────────────────────────────────────────
    if run and query.strip():
        from rag_p4 import run_full_pipeline
        bar = st.progress(0, text="Retrieving chunks …")
        t0  = time.time()
        try:
            bar.progress(20, text="Filtering context …")
            bar.progress(45, text="Generating answer …")
            result = run_full_pipeline(
                query           = query,
                db_path         = DB_PATH,
                sim_threshold   = sim_threshold,
                skip_nli_filter = skip_nli,
                verbose         = False,
            )
            bar.progress(80, text="Scoring …")
            time.sleep(0.1)
            bar.progress(100, text="Done!")
            time.sleep(0.2)
            bar.empty()
            st.session_state.result = result
            st.session_state.history.insert(0, {
                "query":   query,
                "verdict": result.verdict,
                "faith":   result.faithfulness,
                "score":   result.combined_score,
                "time":    round(time.time()-t0, 1),
                "result":  result,
            })
        except Exception as e:
            bar.empty()
            st.error(f"Pipeline error: {e}")
            st.info("Make sure `ollama serve` is running in a separate terminal.")

    # ── Show result ────────────────────────────────────────────────────────────
    if st.session_state.result:
        r = st.session_state.result

        # Verdict banner
        vcfg = {
            "grounded":     ("✓ GROUNDED",     "#1A7A3A","#EAF3DE"),
            "borderline":   ("⚠ BORDERLINE",   "#7A4A00","#FAEEDA"),
            "hallucinated": ("✗ HALLUCINATED", "#7A1A1A","#FCEBEB"),
        }
        lbl, tc, bg = vcfg.get(r.verdict, ("?","#333","#eee"))
        st.markdown(
            f"<div class='verdict-box' style='background:{bg};border-left:5px solid {tc};color:{tc}'>"
            f"{lbl} &nbsp;·&nbsp; <span style='font-weight:400;font-size:15px'>"
            f"combined score: {r.combined_score:.3f}</span></div>",
            unsafe_allow_html=True,
        )

        # 4 score cards
        def card(col, lbl, val, desc, good=0.70):
            col.markdown(
                f"<div class='score-label'>{lbl}</div>"
                f"<div class='score-val' style='color:{'#1A7A3A' if val>=good else '#BA7517' if val>=.5 else '#E24B4A'}'>"
                f"{val:.3f}</div><div class='score-desc'>{desc}</div>",
                unsafe_allow_html=True,
            )

        a,b,c,d = st.columns(4)
        card(a, "Faithfulness",     r.scores.faithfulness,     "Grounded / total sentences",        0.75)
        card(b, "Answer relevance", r.scores.answer_relevance, "Query ↔ answer cosine similarity",  0.65)
        card(c, "Context recall",   r.scores.context_recall,   "Sentences traceable to chunks",      0.65)
        card(d, "Combined",         r.scores.combined,         "Weighted: 0.5 / 0.3 / 0.2",         0.75)

        st.divider()

        # Highlighted answer
        st.markdown("### Answer")
        if r.was_fallback:
            st.warning(r.answer)
        else:
            if r.disclaimer:
                st.warning(f"⚠ {r.disclaimer}")

            html = ""
            for s in r.sentences:
                bg2, tc2, ico = {
                    "grounded":     ("#EAF3DE","#1A7A3A","✓"),
                    "uncertain":    ("#FAEEDA","#7A4A00","⚠"),
                    "hallucinated": ("#FCEBEB","#7A1A1A","✗"),
                }.get(s.verdict, ("#eee","#333","?"))
                html += (
                    f"<span title='{ico} {s.verdict} · conf {s.score:.2f}' "
                    f"style='background:{bg2};color:{tc2};border-radius:4px;"
                    f"padding:2px 5px;margin:2px 1px;display:inline'>{s.sentence}</span> "
                )

            st.markdown(f"<div class='sent-block'>{html}</div>", unsafe_allow_html=True)
            st.markdown(
                "<span style='background:#EAF3DE;color:#1A7A3A;padding:2px 8px;border-radius:4px;font-size:12px'>✓ Grounded</span>&nbsp;"
                "<span style='background:#FAEEDA;color:#7A4A00;padding:2px 8px;border-radius:4px;font-size:12px'>⚠ Uncertain</span>&nbsp;"
                "<span style='background:#FCEBEB;color:#7A1A1A;padding:2px 8px;border-radius:4px;font-size:12px'>✗ Hallucinated</span>"
                "<span style='color:#888;font-size:11px;margin-left:8px'>· Hover any sentence for confidence</span>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Retrieved chunks
        with st.expander(f"📦 Chunks  ({r.raw_chunks} retrieved · {r.kept_after_sim} after sim · {r.kept_after_nli_filter} after NLI)"):
            for i,(doc,score) in enumerate(zip(r.w3_result.w2_result.raw_docs,
                                               r.w3_result.w2_result.raw_scores),1):
                kept = score >= sim_threshold
                src  = os.path.basename(doc.metadata.get("source","?"))
                cls  = "chunk-kept" if kept else "chunk-drop"
                tag  = f"✓ kept (score {score:.3f})" if kept else f"✗ dropped (score {score:.3f} < {sim_threshold})"
                st.markdown(
                    f"<div class='chunk-box {cls}'><b>Chunk {i}</b> · {src} · {tag}<br>"
                    f"<span style='color:#555'>{doc.page_content[:220]}…</span></div>",
                    unsafe_allow_html=True,
                )

        # Per-sentence table
        with st.expander("🔬 Per-sentence NLI breakdown"):
            if r.sentences:
                st.dataframe({
                    "Sentence":   [s.sentence[:80]+("…" if len(s.sentence)>80 else "") for s in r.sentences],
                    "NLI label":  [s.label   for s in r.sentences],
                    "Verdict":    [s.verdict for s in r.sentences],
                    "Confidence": [round(s.score,3) for s in r.sentences],
                }, use_container_width=True)
            else:
                st.caption("No sentences (fallback used).")

        # Re-query notice
        if r.result.requery_attempted:
            if r.result.requery_succeeded:
                st.info("🔄 Auto re-query triggered and succeeded — showing improved answer.")
            else:
                st.error("🔄 Auto re-query triggered but also produced an unreliable answer.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_e:
    st.markdown("### 📊 Benchmark Evaluation")
    st.info("Edit queries to match your uploaded document. Label each as grounded or hallucinated. Then click Run Evaluation.")

    default_eval = [
        {"query":"What is the main contribution of this work?",      "label":"grounded"},
        {"query":"What method or approach does this paper propose?",  "label":"grounded"},
        {"query":"What datasets or experiments were used?",           "label":"grounded"},
        {"query":"What problem does this paper try to solve?",        "label":"grounded"},
        {"query":"What are the main results of this paper?",          "label":"grounded"},
        {"query":"What is the current price of Bitcoin?",             "label":"hallucinated"},
        {"query":"Who won the FIFA World Cup in 2022?",               "label":"hallucinated"},
        {"query":"What is the boiling point of nitrogen?",            "label":"hallucinated"},
        {"query":"Who is the CEO of Apple right now?",                "label":"hallucinated"},
        {"query":"What is the recipe for chocolate cake?",            "label":"hallucinated"},
    ]

    edited = []
    for i,q in enumerate(default_eval):
        ca, cb = st.columns([4,1])
        with ca: qt = st.text_input(f"Query {i+1}", value=q["query"], key=f"eq{i}")
        with cb: lb = st.selectbox("Label",["grounded","hallucinated"],
                                   index=0 if q["label"]=="grounded" else 1, key=f"el{i}")
        edited.append({"query":qt,"label":lb})

    run_eval = st.button("▶ Run Evaluation", type="primary", disabled=not st.session_state.doc_ready)

    if run_eval:
        from rag_p4 import run_full_pipeline
        rows    = []
        bar     = st.progress(0)
        total   = len(edited)

        for i, item in enumerate(edited):
            bar.progress(int((i/total)*100), text=f"Query {i+1}/{total} …")
            try:
                res  = run_full_pipeline(query=item["query"], db_path=DB_PATH,
                                         sim_threshold=sim_threshold,
                                         skip_nli_filter=skip_nli, verbose=False)
                pred = "hallucinated" if res.verdict=="hallucinated" else "grounded"
                rows.append({"Query":item["query"][:55],"Label":item["label"],
                             "Predicted":pred,"Verdict":res.verdict,
                             "Faithfulness":round(res.faithfulness,3),
                             "Combined":round(res.combined_score,3),
                             "Correct":"✓" if pred==item["label"] else "✗"})
            except:
                rows.append({"Query":item["query"][:55],"Label":item["label"],
                             "Predicted":"error","Verdict":"error",
                             "Faithfulness":0,"Combined":0,"Correct":"✗"})

        bar.progress(100); time.sleep(0.2); bar.empty()

        TP=sum(1 for r in rows if r["Predicted"]=="hallucinated" and r["Label"]=="hallucinated")
        FP=sum(1 for r in rows if r["Predicted"]=="hallucinated" and r["Label"]=="grounded")
        FN=sum(1 for r in rows if r["Predicted"]=="grounded"     and r["Label"]=="hallucinated")
        TN=sum(1 for r in rows if r["Predicted"]=="grounded"     and r["Label"]=="grounded")
        P  = TP/(TP+FP) if (TP+FP)>0 else 0
        R  = TP/(TP+FN) if (TP+FN)>0 else 0
        F1 = 2*P*R/(P+R) if (P+R)>0 else 0
        AC = (TP+TN)/total if total>0 else 0

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Precision", f"{P*100:.1f}%",  help="When system says hallucinated, how often correct?")
        m2.metric("Recall",    f"{R*100:.1f}%",  help="Of all actual hallucinations, how many caught?")
        m3.metric("F1 Score",  f"{F1*100:.1f}%", help="Harmonic mean of precision and recall")
        m4.metric("Accuracy",  f"{AC*100:.1f}%", help="Overall correct predictions / total")

        st.divider()
        st.dataframe(rows, use_container_width=True)

        buf = io.StringIO()
        csv.DictWriter(buf, fieldnames=rows[0].keys()).writeheader() or \
            csv.DictWriter(buf, fieldnames=rows[0].keys()).writerows(rows)
        # properly write CSV
        buf2 = io.StringIO()
        w = csv.DictWriter(buf2, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
        st.download_button("⬇ Download CSV", buf2.getvalue(),
                           file_name="eval_results.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_h:
    st.markdown("### 🕘 Query History")
    if not st.session_state.history:
        st.info("No queries yet in this session.")
    else:
        vc = {"grounded":"#1A7A3A","borderline":"#BA7517","hallucinated":"#E24B4A"}
        ve = {"grounded":"✓","borderline":"⚠","hallucinated":"✗"}
        for h in st.session_state.history:
            col = vc.get(h["verdict"],"#333")
            ico = ve.get(h["verdict"],"?")
            st.markdown(
                f"<div class='hist-item'>"
                f"<span style='color:{col};font-weight:600'>{ico} {h['verdict'].upper()}</span>"
                f" · score {h['score']:.3f} · faithfulness {h['faith']:.3f} · {h['time']}s<br>"
                f"<span>{h['query']}</span></div>",
                unsafe_allow_html=True,
            )
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_a:
    st.markdown("### ℹ️ About this system")
    st.markdown("""
**What it does**

Detects whether answers generated by a RAG pipeline are grounded in the
retrieved source document or hallucinated by the LLM from its own training knowledge.

---

**How detection works**

| Stage | Week | What happens |
|---|---|---|
| Document ingestion | 1 | Chunks → embeds → FAISS index |
| Retrieval | 1 | Query embedded → top-k similar chunks via cosine search |
| Similarity filter | 2 | Drops chunks with cosine score < threshold |
| NLI context filter | 2 | Drops chunks labelled neutral/contradiction vs query |
| LLM generation | 1 | Llama 3.2 generates answer from clean context |
| Sentence splitting | 3 | Answer broken into individual sentences |
| NLI detection | 3 | Each sentence checked: grounded / uncertain / hallucinated |
| RAGAS scoring | 4 | Faithfulness + relevance + recall → combined score |
| Verdict | 4 | Threshold → Grounded / Borderline / Hallucinated |
| Auto re-query | 4 | If Hallucinated: rewritten query retried once |

---

**Key papers**

| Paper | Year | arXiv |
|---|---|---|
| Retrieval-Augmented Generation — Lewis et al. | 2020 | 2005.11401 |
| Sentence-BERT — Reimers & Gurevych | 2019 | 1908.10084 |
| TRUE: Factual Consistency Evaluation — Honovich et al. | 2022 | 2204.04991 |
| RAGAS — Es et al. | 2023 | 2309.15217 |
| RAGTruth — Wu et al. | 2024 | 2401.00396 |
| LRP4RAG — Bi et al. | 2024 | 2408.15533 |

---

**Tech stack — 100% free and local**

| Component | Tool |
|---|---|
| LLM | Llama 3.2:3b via Ollama |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| NLI model | cross-encoder/nli-deberta-v3-base |
| UI | Streamlit |
| API key needed | None |
""")