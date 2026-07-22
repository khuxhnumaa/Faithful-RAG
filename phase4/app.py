import streamlit as st
import sys, os, time, csv, io, uuid
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / ".." / "phase3"))
sys.path.insert(0, str(BASE / ".." / "phase2"))
sys.path.insert(0, str(BASE / ".." / "week1"))

from doc_store import DocStore, get_index_path, get_upload_path

db = DocStore()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Hallucination Detector",
    page_icon="/",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
.verdict-box  { padding:14px 20px;border-radius:10px;margin-bottom:16px;
                font-size:18px;font-weight:600; }
.score-label  { font-size:11px;color:#888;text-transform:uppercase;
                letter-spacing:.05em;margin-bottom:4px; }
.score-val    { font-size:28px;font-weight:600;margin-bottom:2px; }
.score-desc   { font-size:11px;color:#888; }
.sent-block   { line-height:2.3;font-size:15px;padding:12px;
                border-radius:8px;background:#fafafa; }
.chunk-box    { background:#f8f9fa;border-left:3px solid #ccc;
                padding:8px 12px;border-radius:4px;margin-bottom:8px;font-size:13px; }
.chunk-kept   { border-left-color:#1A7A3A; }
.chunk-drop   { border-left-color:#E24B4A;opacity:.6; }
.doc-card     { padding:10px 12px;border-radius:8px;border:.5px solid #e0e0e0;
                margin-bottom:6px;background:#fafafa;color:black; }
.doc-card.active { border-color:#1F3066;background:#EEF2FF; }
.hist-item    { padding:10px 14px;border-radius:8px;border:.5px solid #e0e0e0;
                margin-bottom:8px;font-size:13px; }
</style>""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("active_doc_id", None), ("history", []), ("result", None),
              ("confirm_delete", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helper: get active doc ────────────────────────────────────────────────────
def get_active_doc():
    if st.session_state.active_doc_id:
        return db.get_document(st.session_state.active_doc_id)
    docs = db.list_documents()
    if docs:
        st.session_state.active_doc_id = docs[0].id
        return docs[0]
    return None

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## DOCUMENTS")

    # ── Upload new document ────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a new document",
        type=["pdf", "txt"],
        key="uploader",
        help="Saved permanently",
    )

    if uploaded:
        existing = db.document_exists(uploaded.name)

        if existing:
            # Document already in DB — just select it
            st.info(f"'{uploaded.name}' already uploaded. Selecting it.")
            st.session_state.active_doc_id = existing.id
            st.session_state.result = None
            st.rerun()
        else:
            # New document — save file + build FAISS index
            with st.spinner(f"Indexing {uploaded.name} …"):
                try:
                    doc_id     = str(uuid.uuid4())
                    file_path  = get_upload_path(uploaded.name)
                    index_path = get_index_path(doc_id)

                    # Save raw file
                    with open(file_path, "wb") as f:
                        f.write(uploaded.getvalue())

                    # Build FAISS index
                    import tempfile, shutil
                    tmp = tempfile.mkdtemp()
                    shutil.copy(file_path, tmp)

                    from ingest import load_documents, chunk_documents, build_vectorstore
                    docs_loaded = load_documents(tmp)
                    chunks      = chunk_documents(docs_loaded)
                    build_vectorstore(chunks, index_path)
                    shutil.rmtree(tmp, ignore_errors=True)

                    # Save to DB
                    doc = db.save_document(
                        doc_id=doc_id,
                        name=uploaded.name,
                        chunk_count=len(chunks),
                        file_path=file_path,
                        index_path=index_path,
                    )
                    st.session_state.active_doc_id = doc_id
                    st.session_state.result = None
                    st.success(f"✓ {len(chunks)} chunks indexed")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error indexing: {e}")

    st.divider()

    # ── Document list ──────────────────────────────────────────────────────────
    all_docs = db.list_documents()

    if not all_docs:
        st.caption("No documents yet. Upload one above.")
    else:
        st.caption(f"{len(all_docs)} document{'s' if len(all_docs)>1 else ''} stored")

        active = get_active_doc()

        for doc in all_docs:
            is_active = (active and doc.id == active.id)

            # ── Confirm delete dialog ──────────────────────────────────────────
            if st.session_state.confirm_delete == doc.id:
                st.warning(f"Delete **{doc.name}**?")
                ca, cb = st.columns(2)
                with ca:
                    if st.button("Yes, delete", key=f"confirm_{doc.id}",
                                 type="primary", use_container_width=True):
                        db.delete_document(doc.id)
                        if st.session_state.active_doc_id == doc.id:
                            st.session_state.active_doc_id = None
                            st.session_state.result = None
                        st.session_state.confirm_delete = None
                        st.rerun()
                with cb:
                    if st.button("Cancel", key=f"cancel_{doc.id}",
                                 use_container_width=True):
                        st.session_state.confirm_delete = None
                        st.rerun()
                continue

            # ── Document card ──────────────────────────────────────────────────
            col_info, col_menu = st.columns([5, 1])

            with col_info:
                # badge = "🟢" if is_active else "⚪"
                st.markdown(
                    f"<div class='doc-card {'active' if is_active else ''}'>"
                    f"{doc.name}</b><br>"
                    f"<span style='font-size:11px;color:#888'>"
                    f"{doc.short_date} · {doc.chunk_count} chunks</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col_menu:
                # Three-dot menu using selectbox styled as a button
                action = st.selectbox(
                    "⋮",
                    options=["⋮", "Select", "Delete"],
                    key=f"menu_{doc.id}",
                    label_visibility="collapsed",
                )

                if action == "Select":
                    st.session_state.active_doc_id = doc.id
                    st.session_state.result = None
                    st.rerun()

                elif action == "Delete":
                    st.session_state.confirm_delete = doc.id
                    st.rerun()

    st.divider()
    #-----
    # sim_threshold = 0.2
    # skip_nli = True
    # ── Settings ───────────────────────────────────────────────────────────────
    st.markdown("## Settings")
    sim_threshold = st.slider(
        "Similarity threshold", 0.20, 0.10, 0.15, 0.05,
        help="Chunks below this score are dropped. Lower = keep more chunks."
    )
    skip_nli = st.checkbox(
        "Skip NLI context filter",
        value=True,
        help="NLI filter disabled"
    )

    st.divider()
    st.markdown("## Verdict thresholds")
    st.markdown("| Score | Verdict |\n|---|---|\n| ≥ 0.75 | ✓ Grounded |\n| 0.50–0.75 | ⚠ Borderline |\n| < 0.50 | ✗ Hallucinated |")
    st.divider()
    # st.caption("Fully local · No API key · Data stays on your Mac")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# Faithful-RAG")
st.caption("Persistent documents · Ask any question · Sentence-level hallucination detection")

active_doc = get_active_doc()

tab_q, tab_e, tab_h, tab_a = st.tabs(["QUERY", "EVALUATE", "HISTORY", "IDEA"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUERY
# ══════════════════════════════════════════════════════════════════════════════
with tab_q:

    if not active_doc:
        st.info("Upload a document from the sidebar to get started.")
        st.stop()

    # Active doc banner
    st.markdown(
        f"<div style='padding:8px 14px;background:#EEF2FF;border-radius:8px;"
        f"border-left:4px solid #1F3066;margin-bottom:16px;font-size:13px;color:#000000'>"
        f"Active document: <b>{active_doc.name}</b> · "
        f"{active_doc.chunk_count} chunks · {active_doc.short_date}"
        f"</div>",
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Ask anything about your document",
        placeholder="e.g.  What is the work experience of this person?",
    )

    c1, c2 = st.columns([1, 6])
    with c1:
        run = st.button("Detect", type="primary",
                        disabled=not bool(query.strip()),
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
                db_path         = active_doc.index_path,
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
                "doc":     active_doc.name,
                "verdict": result.verdict,
                "faith":   result.faithfulness,
                "score":   result.combined_score,
                "time":    round(time.time()-t0, 1),
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
            "grounded":     ("GROUNDED",     "#1A7A3A", "#EAF3DE"),
            "borderline":   ("BORDERLINE",   "#7A4A00", "#FAEEDA"),
            "hallucinated": ("HALLUCINATED", "#7A1A1A", "#FCEBEB"),
        }
        lbl, tc, bg = vcfg.get(r.verdict, ("?", "#333", "#eee"))
        st.markdown(
            f"<div class='verdict-box' style='background:{bg};"
            f"border-left:5px solid {tc};color:{tc}'>"
            f"{lbl} &nbsp;·&nbsp; "
            f"<span style='font-weight:400;font-size:15px'>"
            f"combined score: {r.combined_score:.3f}</span></div>",
            unsafe_allow_html=True,
        )

        # 4 score cards
        def card(col, lbl, val, desc, good=0.70):
            color = "#1A7A3A" if val >= good else ("#BA7517" if val >= .5 else "#E24B4A")
            col.markdown(
                f"<div class='score-label'>{lbl}</div>"
                f"<div class='score-val' style='color:{color}'>{val:.3f}</div>"
                f"<div class='score-desc'>{desc}</div>",
                unsafe_allow_html=True,
            )

        a, b, c, d = st.columns(4)
        # card(a, "Faithfulness",     r.scores.faithfulness,     "Grounded / total sentences",       0.75)
        # card(b, "Answer relevance", r.scores.answer_relevance, "Query ↔ answer cosine similarity", 0.65)
        # card(c, "Context recall",   r.scores.context_recall,   "Sentences traceable to chunks",     0.65)
        # card(d, "Combined",         r.scores.combined,         "Weighted: 0.5 / 0.3 / 0.2",        0.75)
        card(a, "Faithfulness",     r.scores.faithfulness,     "Grounded / total sentences",       0.50)
        card(b, "Answer relevance", r.scores.answer_relevance, "Query ↔ answer cosine similarity", 0.65)
        card(c, "Context recall",   r.scores.context_recall,   "Sentences traceable to chunks",     0.65)
        card(d, "Combined",         r.scores.combined,         "Weighted: 0.5 / 0.3 / 0.2",        0.50)

        st.divider()

        # Answer with highlighted sentences
        st.markdown("### Answer")
        if r.was_fallback:
            st.warning(r.answer)
        else:
            if r.disclaimer:
                st.warning(f"⚠ {r.disclaimer}")

            html = ""
            for s in r.sentences:
                bg2, tc2, ico = {
                    "grounded":     ("#EAF3DE", "#1A7A3A", "✓"),
                    "uncertain":    ("#FAEEDA", "#7A4A00", "⚠"),
                    "hallucinated": ("#FCEBEB", "#7A1A1A", "✗"),
                }.get(s.verdict, ("#eee", "#333", "?"))
                html += (
                    f"<span title='{ico} {s.verdict} · conf {s.score:.2f}' "
                    f"style='background:{bg2};color:{tc2};border-radius:4px;"
                    f"padding:2px 5px;margin:2px 1px;display:inline'>"
                    f"{s.sentence}</span> "
                )

            st.markdown(f"<div class='sent-block'>{html}</div>", unsafe_allow_html=True)
            st.markdown(
                "<span style='background:#EAF3DE;color:#1A7A3A;padding:2px 8px;"
                "border-radius:4px;font-size:12px'>✓ Grounded</span>&nbsp;"
                "<span style='background:#FAEEDA;color:#7A4A00;padding:2px 8px;"
                "border-radius:4px;font-size:12px'>⚠ Uncertain</span>&nbsp;"
                "<span style='background:#FCEBEB;color:#7A1A1A;padding:2px 8px;"
                "border-radius:4px;font-size:12px'>✗ Hallucinated</span>"
                "<span style='color:#888;font-size:11px;margin-left:8px'>"
                "· Hover any sentence for confidence</span>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Retrieved chunks
        with st.expander(
            f"Chunks  ({r.raw_chunks} retrieved · "
            f"{r.kept_after_sim} after sim · "
            f"{r.kept_after_nli_filter} after NLI)"
        ):
            for i, (doc, score) in enumerate(
                zip(r.w3_result.w2_result.raw_docs,
                    r.w3_result.w2_result.raw_scores), 1
            ):
                kept = score >= sim_threshold
                src  = os.path.basename(doc.metadata.get("source", "?"))
                cls  = "chunk-kept" if kept else "chunk-drop"
                tag  = (f"✓ kept (score {score:.3f})" if kept
                        else f"✗ dropped (score {score:.3f} < {sim_threshold})")
                st.markdown(
                    f"<div class='chunk-box {cls}'>"
                    f"<b>Chunk {i}</b> · {src} · {tag}<br>"
                    f"<span style='color:#555'>{doc.page_content[:220]}…</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Per-sentence table
        with st.expander("Per-sentence NLI breakdown"):
            if r.sentences:
                st.dataframe({
                    "Sentence":   [s.sentence[:80]+("…" if len(s.sentence)>80 else "")
                                   for s in r.sentences],
                    "NLI label":  [s.label   for s in r.sentences],
                    "Verdict":    [s.verdict for s in r.sentences],
                    "Confidence": [round(s.score, 3) for s in r.sentences],
                }, use_container_width=True)
            else:
                st.caption("No sentences (fallback used).")

        if r.result.requery_attempted:
            if r.result.requery_succeeded:
                st.info("Auto re-query triggered and succeeded.")
            else:
                st.error("Auto re-query triggered but also failed.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_e:
    st.markdown("### Benchmark Evaluation")

    if not active_doc:
        st.info("Select a document from the sidebar first.")
    else:
        st.info(
            f"Running against: **{active_doc.name}**. "
            "Edit queries to match your document, then click Run."
        )

        default_eval = [
            {"query": "What is the main contribution of this work?",     "label": "grounded"},
            {"query": "What method or approach does this paper propose?", "label": "grounded"},
            {"query": "What datasets or experiments were used?",          "label": "grounded"},
            {"query": "What problem does this paper try to solve?",       "label": "grounded"},
            {"query": "What are the main results of this paper?",         "label": "grounded"},
            {"query": "What is the current price of Bitcoin?",            "label": "hallucinated"},
            {"query": "Who won the FIFA World Cup in 2022?",              "label": "hallucinated"},
            {"query": "What is the boiling point of nitrogen?",           "label": "hallucinated"},
            {"query": "Who is the CEO of Apple right now?",               "label": "hallucinated"},
            {"query": "What is the recipe for chocolate cake?",           "label": "hallucinated"},
        ]

        edited = []
        for i, q in enumerate(default_eval):
            ca, cb = st.columns([4, 1])
            with ca:
                qt = st.text_input(f"Query {i+1}", value=q["query"], key=f"eq{i}")
            with cb:
                lb = st.selectbox("Label", ["grounded", "hallucinated"],
                                  index=0 if q["label"] == "grounded" else 1,
                                  key=f"el{i}")
            edited.append({"query": qt, "label": lb})

        if st.button("Run Evaluation", type="primary"):
            from rag_p4 import run_full_pipeline
            rows  = []
            bar   = st.progress(0)
            total = len(edited)

            for i, item in enumerate(edited):
                bar.progress(int((i/total)*100), text=f"Query {i+1}/{total} …")
                try:
                    res  = run_full_pipeline(
                        query=item["query"],
                        db_path=active_doc.index_path,
                        sim_threshold=sim_threshold,
                        skip_nli_filter=skip_nli,
                        verbose=False,
                    )
                    pred = "hallucinated" if res.verdict == "hallucinated" else "grounded"
                    rows.append({
                        "Query":        item["query"][:55],
                        "Label":        item["label"],
                        "Predicted":    pred,
                        "Faithfulness": round(res.faithfulness, 3),
                        "Combined":     round(res.combined_score, 3),
                        "Correct":      "✓" if pred == item["label"] else "✗",
                    })
                except:
                    rows.append({
                        "Query": item["query"][:55], "Label": item["label"],
                        "Predicted": "error", "Faithfulness": 0,
                        "Combined": 0, "Correct": "✗",
                    })

            bar.progress(100); time.sleep(0.2); bar.empty()

            TP = sum(1 for r in rows if r["Predicted"]=="hallucinated" and r["Label"]=="hallucinated")
            FP = sum(1 for r in rows if r["Predicted"]=="hallucinated" and r["Label"]=="grounded")
            FN = sum(1 for r in rows if r["Predicted"]=="grounded"     and r["Label"]=="hallucinated")
            TN = sum(1 for r in rows if r["Predicted"]=="grounded"     and r["Label"]=="grounded")
            P  = TP/(TP+FP) if (TP+FP)>0 else 0
            R  = TP/(TP+FN) if (TP+FN)>0 else 0
            F1 = 2*P*R/(P+R) if (P+R)>0 else 0
            AC = (TP+TN)/total if total>0 else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", f"{P*100:.1f}%")
            m2.metric("Recall",    f"{R*100:.1f}%")
            m3.metric("F1 Score",  f"{F1*100:.1f}%")
            m4.metric("Accuracy",  f"{AC*100:.1f}%")

            st.divider()
            st.dataframe(rows, use_container_width=True)

            buf = io.StringIO()
            w   = csv.DictWriter(buf, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
            st.download_button("⬇ Download CSV", buf.getvalue(),
                               file_name="eval_results.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_h:
    st.markdown("###Query History")
    if not st.session_state.history:
        st.info("No queries yet in this session.")
    else:
        vc = {"grounded": "#1A7A3A", "borderline": "#BA7517", "hallucinated": "#E24B4A"}
        ve = {"grounded": "✓", "borderline": "⚠", "hallucinated": "✗"}
        for h in st.session_state.history:
            col = vc.get(h["verdict"], "#333")
            ico = ve.get(h["verdict"], "?")
            st.markdown(
                f"<div class='hist-item'>"
                f"<span style='color:{col};font-weight:600'>{ico} {h['verdict'].upper()}</span>"
                f" · score {h['score']:.3f} · faithfulness {h['faith']:.3f}"
                f" · {h['time']}s · 📄 {h['doc']}<br>"
                f"<span>{h['query']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_a:
    st.markdown("### About this system")

    if active_doc:
        st.markdown(f"**Active document:** {active_doc.name}  ·  {active_doc.chunk_count} chunks  ·  {active_doc.short_date}")
        st.divider()

    total_docs = db.count()
    st.markdown(f"**Documents stored:** {total_docs}")
    st.markdown(f"**Storage location:** `{BASE / 'data'}`")

    st.markdown("""
---
**Detection pipeline**

| Stage | Week | What happens |
|---|---|---|
| Upload + ingest | 1 | File saved · chunks embedded · FAISS index stored per document |
| Retrieval | 1 | Query embedded → top-k similar chunks via cosine search |
| Similarity filter | 2 | Drops chunks with cosine score < threshold |
| LLM generation | 1 | Llama 3.2 generates answer from filtered context |
| Sentence splitting | 3 | Answer broken into individual sentences |
| NLI detection | 3 | Each sentence: grounded / uncertain / hallucinated |
| RAGAS scoring | 4 | Faithfulness + relevance + recall → combined score |
| Verdict | 4 | Threshold → Grounded / Borderline / Hallucinated |
| Auto re-query | 4 | If Hallucinated: rewritten query retried once |

---
**Key papers**

| Paper | arXiv |
|---|---|
| RAG — Lewis et al., 2020 | 2005.11401 |
| Sentence-BERT — Reimers & Gurevych, 2019 | 1908.10084 |
| TRUE — Honovich et al., 2022 | 2204.04991 |
| RAGAS — Es et al., 2023 | 2309.15217 |
| RAGTruth — Wu et al., 2024 | 2401.00396 |

---
**Tech stack — 100% free and local**

Llama 3.2 via Ollama · sentence-transformers · FAISS · nli-deberta · SQLite · Streamlit
""")