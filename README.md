# Redrob Hackathon — Candidate Ranking Pipeline

Ranking pipeline for the **Founding Senior AI Engineer** role. Produces a top-100 submission CSV from the released `candidates.jsonl` pool.

## Guiding philosophy

> *Eliminate only candidates who are extremely unlikely to be selected according to the JD. Everything else is expressed as preferences through scoring.*

## Approach

The pipeline runs in six stages:

- **Stage 0:** one-pass feature engineering over the full pool.
- **Stage 1:** cheap direct eliminations (location, no AI/ML evidence, extreme job-hopping, consulting-only, non-tech career, pure research).
- **Stage 2:** weighted sub-scores (JD alignment, experience, behavioral, production, like-to-have, location) + BM25/TF-IDF lexical retrieval over all survivors.
- **Stage 2D:** adaptive shortlist size chosen by elbow + percentile guard, then dense retrieval with a FAISS `IndexFlatIP` over the shortlist using the **pre-computed full-pool BGE embeddings**, followed by cross-encoder re-rank of the top head.
- **Stage 3:** honeypot detection with hard-reject threshold + soft penalty for borderline cases.
- **Stage 4:** concise, candidate-specific reasoning generation for the top 100.

The expensive full-pool BGE embedding step is done **once offline** via `get_embeddings.py`; the ranking step is CPU-only and finishes in ~2 minutes.

## Precomputation

Precomputation must be completed **before** the ranking step. `get_embeddings.py` is the only precomputation entry point:

1. Check if `cache/bge_embeddings_completed.npz` exists.
2. If not, download it from HuggingFace.
3. If download fails, generate the cache locally from `data/candidates.jsonl`.

Generation is checkpointed every 5,000 candidates so progress is not lost. If [pipeline/rank.py](cci:7://file:///d:/Desktop/India_runs_data_and_ai_challenge/pipeline/rank.py:0:0-0:0) is run without a precomputed cache, it automatically obtains the cache by calling [pipeline/get_embeddings.py](cci:7://file:///d:/Desktop/India_runs_data_and_ai_challenge/pipeline/get_embeddings.py:0:0-0:0) (it first tries to download the pre-computed cache from the project's HuggingFace dataset, which was uploaded during development, then falls back to local generation) and prints a warning that total runtime will be significantly higher.

**Embedding model:** `BAAI/bge-small-en-v1.5` (384-dim, 512-token, CPU-friendly).

```bash
python get_embeddings.py
```

## Selection methodology

This section documents exactly what the pipeline considers and how each factor is weighted.

### JD preprocessing before retrieval

The job description is read from `jd/job_description_crux.md` (falling back to `jd/job_description.md`). The markdown is parsed into three bullet lists:

- **Must-have** (`Things you absolutely need`)
- **Like-to-have** (`Things we'd like you to have`)
- **Do-not-want** (`Things we explicitly do NOT want`)

A curated technical seed (`TECH_SEED`) is prepended to the query to ground it with retrieval vocabulary. The final query representations are:

- **JD_QUERY:** `TECH_SEED + must_have (×2) + like_have`, lowercased. Repeating must-haves gives them ~2× weight in lexical and dense signals.
- **JD_NEGATIVE:** `do_not_want` bullets + explicit off-domain vocabulary (CV, speech, robotics, consulting, pure research).
- **JD_CE_QUERY:** a natural-language sentence used by the cross-encoder (MS-MARCO style, not keyword-stuffed).

For BGE dense retrieval, the query is prefixed with `Represent this sentence for searching relevant passages: `; cached candidate passages are **not** prefixed.

If the markdown cannot be parsed, hard-coded fallback strings are used.

### Resume preprocessing before embedding

Each candidate profile is lowercased and aggregated into the following fields before embedding:

1. `skills_text` — skill names joined by spaces.
2. `current_title`
3. `summary`
4. `titles_all` — current title + past titles, joined by `|`.
5. `headline`
6. `career_text` — all career descriptions joined.

The final `embed_text` is built by:
- Moving JD-relevant skills to the front of the skills list (using `IR_CORE_KEYWORDS`, `EVAL_KEYWORDS`, `LLM_KEYWORDS`, and `AIML_SKILLS`).
- Deduplicating words after the first occurrence.
- Capping the text at **400 words** (`EMBED_MAX_WORDS = 400`). The BGE encoder itself is limited to **512 tokens**.

An MD5 hash of the final `embed_text` is stored in the cache so stale rows are re-encoded automatically. The same `_build_embed_text` logic exists in both `get_embeddings.py` and `rank.py`, so the cache is compatible with the ranking step.

### Stage 1: Direct eliminations

A candidate is rejected if **any** of the following rules fire:

| Rule | Condition |
| --- | --- |
| **Location** | Not in a preferred hub AND `willing_to_relocate == False`. |
| **No AI/ML evidence** | No relevant skill, no relevant title, and no career-description evidence. |
| **Extreme job-hopping** | ≥ 3 companies and > 70% of jobs shorter than 18 months. |
| **Consulting-only** | 100% of career months at consulting firms. |
| **Entire-career non-tech** | No job passes the tech whitelist. |
| **Pure research without production** | 100% research months and no production evidence. |



> `AIML_SKILLS` is intentionally broad so that relevant AI/ML work is never eliminated, while the scoring weights differentiate the exact JD fit.

### Stage 2: Candidate scoring

Final weights are derived from the JD rubric: `weight ∝ decisiveness × discriminativeness`, normalized to sum to exactly 1.0. The values below are the exact 4-decimal weights used by `rank.py` and the notebook.

| Component | Decisiveness | Discriminativeness | Final Weight |
| --- | --- | --- | --- |
| **JD alignment** | 5 | 5 | **0.3424** |
| **Experience** | 4 | 4 | **0.2192** |
| **Behavioral** | 3 | 4 | **0.1644** |
| **Production** | 5 | 2 | **0.1370** |
| **Like-to-have** | 2 | 3 | **0.0822** |
| **Location** | 2 | 2 | **0.0548** |

The final score is `base_score × penalty_mult`, where `penalty_mult` compounds the off-domain penalty (0.65 / 0.85 / 1.00) and the relocation penalty (0.85 for outside-India candidates).

#### A. Location score

| Condition | Score |
| --- | --- |
| Pune / Noida | 1.00 |
| Delhi NCR / Mumbai / Hyderabad | 0.95 |
| India + willing to relocate | 0.80 |
| India + relocation unknown | 0.60 |
| Outside India + willing to relocate | 0.40 |
| Outside India + unwilling | 0.20 |

#### B. Behavioral score

```
behavioral = 0.28·open_to_work + 0.27·recency + 0.18·recruiter_response_rate
           + 0.12·interview_completion_rate + 0.15·notice_period_score
```

- **Recency:** `exp(-days_since_active / 120)`.
- **Notice period:** ≤30 days = 1.00, 31–60 = 0.85, 61–90 = 0.60, 91–120 = 0.30, >120 = 0.10.

#### C. Experience score

```
experience = 0.5·applied_AI_score + 0.3·total_experience_score + 0.2·career_trajectory_score
```

**Applied AI years (peak 4–5 years):**

| Years | Score |
| --- | --- |
| < 1 | 0.00 |
| < 2 | 0.20 |
| < 3 | 0.40 |
| < 4 | 0.70 |
| 4–5 | 1.00 |
| 6–7 | 0.90 |
| 8–10 | 0.75 |
| > 10 | 0.55 |

**Total experience (peak 5–9 years):**

| Years | Score |
| --- | --- |
| < 3 | 0.00 |
| 3–4 | 0.60 |
| 5–9 | 1.00 |
| 10–15 | 0.80 |
| > 15 | 0.60 |

**Career trajectory score:**
- Promotion trend: `0.5 + 0.25 × seniority_delta`, clipped to [0, 1].
- Current seniority altitude: intern=0.10, junior=0.30, mid=0.50, senior=0.80, lead=0.95, head=1.00.
- Tenure stability: healthy 1–6 year stints = 1.00, very long single stint = 0.80, < 1 year = 0.30, unknown = 0.50.
- Job-hopping haircut: stability is multiplied by `1 - 0.5 × short_switch_ratio`.

#### D. JD alignment score

Phase 1 (over all survivors):
```
jd_align = 0.45·BM25 + 0.30·TF-IDF + 0.25·keyword_evidence
```
(all min-max normalized). If BM25 or TF-IDF is unavailable, the remaining lexical weights are renormalized so they still sum to 1.0.

Phase 2 (over the shortlist):
```
jd_align_v2 = 0.50·dense + 0.25·BM25 + 0.10·TF-IDF + 0.15·keyword_evidence
```
where `dense = positive_cosine - 0.5 × negative_cosine`.

**Keyword evidence lexicons:**
- **IR core:** retrieval, ranking, recommendation, relevance, search, embedding, embeddings, semantic search, vector, bm25, faiss, learning to rank, recommender, matching.
- **Evaluation:** ndcg, mrr, map, evaluation, offline, online, benchmark, a/b, ab test, ab testing, precision, recall.
- **LLM integration:** prompt engineering, fine-tun, lora, qlora, peft, llm, rag, large language model, transformer, deployment.
- **Career evidence (used for the "No AI/ML evidence" filter):** retrieval, ranking, recommendation, relevance, search, embedding, embeddings, semantic, vector, nlp, language model, recommender, personalization, matching, evaluation, deployed ml, ml system, ml pipeline, machine learning.

#### E. Production evidence score

```
production = clip(count_kw(career_text + summary, PRODUCTION_KEYWORDS) / 4.0, 0, 1)
```

**Production keywords:** production, deployed, deploy, scaled, scale, monitoring, latency, pipeline, pipelines, users, serving, real-time, real time, throughput, uptime, sla.

#### F. Like-to-have bonus score

```
like_to_have = (number of matching groups) / (total groups)
```

**Bonus groups:**
- Fine-tuning: lora, qlora, peft, fine-tun.
- Learning-to-rank: lambdamart, xgboost ranker, learning to rank, neural rank.
- HR-tech: recruitment, talent matching, hiring, hr-tech, hr tech, recruiting, marketplace.
- Distributed systems: inference optimization, distributed serving, distributed systems, model serving, triton.
- Open source: open source, open-source, github, contributor, maintainer.

#### G. Off-domain penalty

Applied only if IR evidence is low while CV/speech/robotics signals are high:

| Condition | Penalty | Reason |
| --- | --- | --- |
| IR = 0 and off-domain ≥ 2 | 0.65 | Off-domain focus, no IR evidence. |
| IR = 0 and off-domain = 1 | 0.85 | Limited IR evidence, some off-domain focus. |
| Otherwise | 1.00 | No penalty. |

**Off-domain keywords:**
- CV: yolo, opencv, segmentation, object detection, cnn, image classification, computer vision, gans, diffusion models.
- Speech: asr, wav2vec, audio, tts, speech recognition, speech.
- Robotics: ros, slam, motion planning, robotics, lidar.

#### H. Relocation penalty

Outside-India candidates (score_location < 0.5) receive an additional 0.85 multiplicative haircut.

### Stage 2D: Dense retrieval and re-ranking

1. **Adaptive shortlist:** K is chosen by the larger of (a) the elbow of the sorted score curve and (b) the 95th-percentile score guard. K is clipped between `SHORTLIST_MIN = 500` and `SHORTLIST_MAX = 4000`.
2. **FAISS index:** `IndexFlatIP` is built over the shortlisted contenders from the pre-computed full-pool cache. If FAISS is unavailable, the code falls back to a NumPy dot-product.
3. **Dense scoring:** positive relevance = cosine to `JD_QUERY`; negative relevance = cosine to `JD_NEGATIVE`; final dense signal = `positive - 0.5 × negative`. If any shortlisted contender is missing from the cache, it is removed from the dense shortlist and the pipeline continues with lexical-only scores for those rows.
4. **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80 MB, CPU-friendly) re-ranks the top 400 contenders; final JD alignment for those rows is `0.75·cross_encoder + 0.25·bi_encoder`. If the cross-encoder is unavailable, the bi-encoder score is kept.

### Stage 3: Honeypot detection

Four suspicion signals are scored:

| Signal | Points | Condition |
| --- | --- | --- |
| Expert skills with ~0 months | 2–3 | 2 expert skills with 0 months = 2 pts; ≥3 = 3 pts. |
| Implausible expert breadth | 2 | ≥10 expert skills. |
| Experience/tenure mismatch | 1–3 | Gap > 5 years = 1 pt; gap > 8 years = 3 pts. |
| Title inflation | 2 | Senior title token with < 2 years total tenure. |

**Thresholds:**
- ≥ 3 points → hard reject.
- 1–2 points → soft penalty (final score × 0.90).
- 0 points → clean.

The top 100 must contain fewer than 10% honeypots.

### Stage 4: Reasoning generation

Reasoning is generated only for the top 100 and includes:
- Title + years of experience (with applied AI/ML years if > 0).
- Top 3 matching JD skills.
- One career highlight.
- Behavioral signals (open to work, response rate, recently active).
- Notice period.
- Concerns if any (notice period > 60 days, location/visa risk, short tenure, low response rate, penalty reason, honeypot flag).
- Skill note only when other concerns exist **and** the full profile (not just skills) lacks explicit evaluation/ranking metrics (NDCG/MRR/MAP/A-B).
- Strongest scoring axis only when it is **not** JD alignment.

## What the pipeline does not consider

### Protected / sensitive attributes

Gender, age, date of birth, ethnicity, religion, marital status, disability, salary, expected salary range, candidate photo, anonymized name.

### Unused structured profile fields

- `education` (degree, institution, field of study, grade, tier)
- `certifications`
- `languages`
- `current_company_size`
- Career-history `is_current`, `industry`, `company_size`
- Skill `endorsements`

### Unused Redrob signals

`profile_completeness_score`, `signup_date`, `profile_views_received_30d`, `applications_submitted_30d`, `avg_response_time_hours`, `skill_assessment_scores`, `connection_count`, `endorsements_received`, `expected_salary_range_inr_lpa`, `preferred_work_mode`, `github_activity_score`, `search_appearance_30d`, `saved_by_recruiters_30d`, `offer_acceptance_rate`, `verified_email`, `verified_phone`, `linkedin_connected`.

### Other exclusions

- No external API or LLM is used during ranking.
- No human-in-the-loop or non-deterministic decision-making.
- No hard filters based on years of experience alone; experience is scored, not eliminated.

## Repository layout

This folder (`pipeline/`) is self-contained and contains everything needed to run the ranking pipeline or the Streamlit sandbox.

- `rank.py` — single-command entry point that produces `output/submission.csv`.
- `app.py` — Streamlit sandbox.
- `ranking_pipeline_final.ipynb` — development notebook with the full pipeline and visualizations.
- `get_embeddings.py` — precomputation: downloads or generates the full-pool BGE embedding cache.
- `requirements.txt` — Python dependencies.
- `submission_metadata.yaml` — portal metadata (fill before submitting).
- `data/` — candidate pool (`candidates.jsonl`, `candidates_small.jsonl`).
- `cache/` — pre-computed full-pool BGE embeddings (`bge_embeddings_completed.npz`).
- `jd/` — job description files (`job_description_crux.md`, `job_description.md`).
- `validator/` — official submission validator (`validate_submission.py`).
- `output/` — generated submission (`submission.csv`).
- `../organizer/` — challenge bundle provided by the organizers.

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/GuranshGoyal/RedRob26.git
cd RedRob26

# 2. Install Git LFS and pull the .npz embedding cache
#    (Git LFS is required because the embedding cache exceeds GitHub's 100 MB limit.)
git lfs install
git lfs pull

# 3. Create a virtual environment (recommended)
python -m venv venv

# Activate (Windows Git Bash / PowerShell)
source venv/Scripts/activate

# Activate (Linux/macOS)
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Ensure the full-pool BGE embedding cache is present
#    The cache is stored in Git LFS. If it is missing, get_embeddings.py can
#    download it from HuggingFace (~1–2 minutes) or generate it locally (~300 minutes).
cd pipeline
python get_embeddings.py
```

`get_embeddings.py` first checks for the pre-computed embedding cache. If it is missing, it tries to download from the project's HuggingFace dataset (uploaded during development, ~1–2 minutes). If that fails, it falls back to generating the cache locally from `data/candidates.jsonl` (~300 minutes on the reference machine).
 

## Reproduce the submission
 
Precomputation must be completed **before** the ranking step. The ranking step makes no network calls and finishes in ≤ 5 minutes.

```bash
# Single-step command to run the ranking pipeline
cd pipeline
python rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv
```

Expected ranking runtime: **~2 minutes** on a modern CPU (≤ 5 minutes on the competition sandbox).

## Validate the output

```bash
cd pipeline
python validator/validate_submission.py output/submission.csv
```

## Compute constraints

- **Runtime:** ranking step ≤ 5 minutes wall-clock.
- **Memory:** ≤ 16 GB RAM.
- **CPU only:** no GPU used during ranking.
- **No network:** ranking step makes no external API calls.
- **Disk:** pre-computed embeddings + models ≤ 5 GB.

## Sandbox

A Streamlit sandbox is available at: `https://resumegs.streamlit.app/`.