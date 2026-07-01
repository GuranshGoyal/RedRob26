import os
import tempfile
import time
from pathlib import Path

import streamlit as st

from rank import run_pipeline

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯")

st.title("🎯 Redrob Candidate Ranker")
st.markdown(
    "Streamlit sandbox for the **Founding Senior AI Engineer** ranking pipeline. "
    "Upload a small `candidates.jsonl` sample and run the full pipeline end-to-end."
)

uploaded_file = st.file_uploader("Upload candidates.jsonl (small sample)", type=["jsonl", "json", "gz"])

if uploaded_file is not None:
    # Save upload to a temp file
    suffix = ".jsonl.gz" if uploaded_file.name.endswith(".gz") else ".jsonl"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        input_path = tmp.name

    out_path = input_path.replace(suffix, "_submission.csv")

    st.write(f"Input: `{uploaded_file.name}` ({len(uploaded_file.getvalue()) / 1024:.1f} KB)")

    if st.button("Run ranking pipeline"):
        with st.spinner("Running pipeline... (ranking step target ≤ 5 min)"):
            t0 = time.time()
            run_pipeline(input_path, out_path)
            elapsed = time.time() - t0

        st.success(f"Pipeline finished in {elapsed:.1f}s")

        if os.path.exists(out_path):
            import pandas as pd

            df = pd.read_csv(out_path)
            st.write(f"Output: `{len(df)}` rows")

            if len(df) < 100:
                st.warning(
                    "Sandbox sample produced a partial ranking (fewer than 100 rows). "
                    "This is expected for small samples; the final submission must use the full dataset."
                )
            else:
                st.success("Full 100-row submission generated.")

            st.dataframe(df, use_container_width=True)

            st.download_button(
                label="Download submission.csv",
                data=Path(out_path).read_bytes(),
                file_name="submission.csv",
                mime="text/csv",
            )
        else:
            st.error("Output file was not generated.")

    st.markdown("---")
    st.markdown("### About this pipeline")
    st.markdown(
        "- **Stage 1:** hard-elimination filters over 100k candidates.\n"
        "- **Stage 2:** BM25 + TF-IDF + weighted sub-scores over all survivors.\n"
        "- **Stage 2D:** adaptive shortlist → FAISS dense retrieval → cross-encoder re-rank.\n"
        "- **Stage 3:** honeypot detection with hard-reject threshold.\n"
        "- **Stage 4:** concise, data-driven reasoning generation for top 100.\n\n"
        "Full-pool BGE embeddings are pre-computed offline; the sandbox only runs the ranking step."
    )
else:
    st.info("Upload a small JSONL sample to get started.")
