#!/usr/bin/env python3
# Redrob Hackathon — Candidate Ranking Pipeline
# Single-command entry point that reproduces submission.csv from candidates.jsonl.
# Requires the pre-computed full-pool BGE embedding cache to be built first:
#     python get_embeddings.py
# This script performs only the ranking step; no precomputation happens here.

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse


def run_pipeline(candidates_path="candidates.jsonl", output_path="submission.csv"):
    """Execute the full ranking pipeline and write submission.csv."""
    # Override the notebook globals with CLI arguments
    global DATA_PATH, OUTPUT_CSV
    DATA_PATH = candidates_path
    OUTPUT_CSV = output_path

    # ===== cell 2118051c =====
    # ---- Force a PyTorch-only backend BEFORE any transformers import (idempotent) ----
    import os
    os.environ.setdefault("USE_TF", "0")                       # skip TensorFlow in transformers
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import re
    import sys
    import time
    import json
    import gzip
    import warnings
    from datetime import datetime

    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", 60)

    # ----------------------------- RUN CONFIG ---------------------------------
    TOP_N      = 100

    # --- Bi-encoder (dense retrieval) ---
    EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, 512-token, strong CPU retrieval model
    EMBED_MAX_TOKENS = 512                    # bge handles 512 (2x MiniLM) -> less truncation
    # bge retrieval works best when the QUERY (the JD) carries this instruction; passages get none.
    BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
    USE_DENSE_EMBEDDINGS = True              # auto-falls back to BM25+TF-IDF if unavailable

    # --- Cross-encoder (precise re-rank of the top head) ---
    USE_CROSS_ENCODER   = True              # auto-falls back to bi-encoder ranking if unavailable
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # ~80MB, CPU-friendly
    CE_TOPK             = 400               # only the top-CE_TOPK shortlist rows are re-ranked

    # --- Offline dense index (embeddings precomputed & cached -> full-contender retrieval) ---
    EMB_CACHE_PATH = "cache/bge_embeddings_completed.npz"    # persisted BGE vectors (candidate_id + text-hash keyed)
    USE_FAISS      = True                     # FAISS IndexFlatIP retrieval; falls back to numpy dot

    # Adaptive-shortlist controls (the SIZE itself is computed in Stage 2D, not fixed here).
    SHORTLIST_MIN          = 5 * TOP_N       # floor: enough recall for a 100-pick re-rank
    SHORTLIST_MAX          = 4000            # ceiling so the dense step stays inside the CPU budget
    SHORTLIST_TIME_BUDGET_S = 120            # seconds we allow the dense encode to take

    # ------------------- VERIFY PRE-COMPUTED EMBEDDING CACHE ------------------
    if USE_DENSE_EMBEDDINGS and not os.path.exists(EMB_CACHE_PATH):
        raise FileNotFoundError(
            f"Embedding cache not found at {EMB_CACHE_PATH}. "
            "Run the precomputation step first: python get_embeddings.py"
        )

    # ------------------- DERIVE WEIGHTS FROM THE JD RUBRIC --------------------
    # weight ∝ decisiveness × discriminativeness  (see markdown above)
    _RUBRIC = {
        # component        decisiveness  discriminativeness
        "jd_alignment":  (5, 5),
        "experience":    (4, 4),
        "behavioral":    (3, 4),
        "production":    (5, 2),
        "like_to_have":  (2, 3),
        "location":      (2, 2),
    }
    _raw = {k: dec * disc for k, (dec, disc) in _RUBRIC.items()}
    _tot = sum(_raw.values())
    WEIGHTS = {k: round(v / _tot, 4) for k, v in _raw.items()}
    # fix rounding drift so weights sum to exactly 1.0
    WEIGHTS["jd_alignment"] += round(1.0 - sum(WEIGHTS.values()), 4)

    weight_table = pd.DataFrame(
        [(k, d, s, d * s, WEIGHTS[k]) for k, (d, s) in _RUBRIC.items()],
        columns=["component", "decisiveness", "discriminativeness", "raw", "weight"],
    ).sort_values("weight", ascending=False).reset_index(drop=True)

    print("Derived scoring weights (sum = %.4f):" % sum(WEIGHTS.values()))
    print(weight_table.to_string(index=False))
    # ===== cell 15d7536f =====
    # ============================ LEXICONS ====================================
    # --- Locations (preferred hubs from the JD) ---
    PREFERRED_PRIMARY   = ["pune", "noida"]                                  # location score 1.00
    PREFERRED_SECONDARY = ["delhi", "ncr", "gurgaon", "gurugram",
                           "mumbai", "navi mumbai", "hyderabad"]             # location score 0.95
    PREFERRED_LOCATIONS = PREFERRED_PRIMARY + PREFERRED_SECONDARY

    # --- Relevant AI/ML SKILLS (mapped from the 133 unique dataset skills + JD) ---
    AIML_SKILLS = {
        # retrieval / search / ranking (the JD core)
        "information retrieval", "information retrieval systems", "semantic search", "vector search",
        "recommendation systems", "ranking systems", "learning to rank", "bm25", "faiss", "pinecone",
        "weaviate", "milvus", "qdrant", "pgvector", "elasticsearch", "opensearch", "haystack",
        "search backend", "search infrastructure", "search & discovery", "indexing algorithms",
        "content matching", "vector representations", "text encoders",
        # embeddings / nlp / llm
        "embeddings", "sentence transformers", "hugging face transformers", "llms", "llm", "rag",
        "langchain", "llamaindex", "prompt engineering", "fine-tuning llms", "lora", "qlora", "peft",
        "nlp", "natural language processing", "model adaptation",
        # core ml
        "machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn", "data science",
        "feature engineering", "statistical modeling", "reinforcement learning", "time series",
        "forecasting", "model adaptation",
        # mlops / serving
        "mlops", "mlflow", "kubeflow", "bentoml", "weights & biases", "open-source ml libraries",
    }

    # --- Relevant TITLES (current or past) ---
    RELEVANT_TITLES = [
        "ml engineer", "machine learning engineer", "ai engineer", "applied scientist",
        "applied ai", "data scientist", "research engineer", "nlp engineer", "search engineer",
        "relevance engineer", "recommendation engineer", "ranking engineer", "mlops engineer",
        "ml scientist", "research scientist", "deep learning", "computer vision engineer",
        # data-adjacent whitelist (per plan §5)
        "data engineer", "analytics engineer", "data analyst", "bi engineer",
        "solutions architect", "software engineer", "backend engineer", "platform engineer",
    ]

    # --- Seniority ladder (for career-trajectory / promotion-trend scoring) ---
    # Each role title is mapped to an ordinal rank; a plain IC role defaults to mid-level (3).
    SENIORITY_LADDER = [
        (["intern", "trainee", "apprentice"], 1),
        (["junior", "jr ", "associate", "graduate", "entry-level", "entry level"], 2),
        (["senior", "sr ", "sr.", "specialist", "lead engineer"], 4),
        (["lead", "principal", "staff", "manager", "architect"], 5),
        (["head", "director", "vp", "vice president", "chief", "cto", "founder", "co-founder"], 6),
    ]
    def seniority_rank(title):
        matched = [lvl for kws, lvl in SENIORITY_LADDER if any(k in title for k in kws)]
        return max(matched) if matched else 3   # default: mid-level individual contributor

    # --- Career-description EVIDENCE of relevant work (plan §2) ---
    CAREER_EVIDENCE = [
        "retrieval", "ranking", "recommendation", "relevance", "search", "embedding", "embeddings",
        "semantic", "vector", "nlp", "language model", "recommender", "personalization",
        "matching", "evaluation", "deployed ml", "ml system", "ml pipeline", "machine learning",
    ]

    # --- IR / retrieval core (for D + penalty G) ---
    IR_CORE_KEYWORDS = [
        "retrieval", "ranking", "recommendation", "relevance", "search", "embedding", "embeddings",
        "semantic search", "vector", "bm25", "faiss", "learning to rank", "recommender", "matching",
    ]

    # --- Evaluation signals (JD: NDCG/MRR/MAP/offline/online/A-B) ---
    EVAL_KEYWORDS = ["ndcg", "mrr", "map", "evaluation", "offline", "online", "benchmark",
                     "a/b", "ab test", "ab testing", "precision", "recall"]

    # --- LLM integration evidence ---
    LLM_KEYWORDS = ["prompt engineering", "fine-tun", "lora", "qlora", "peft", "llm", "rag",
                    "large language model", "transformer", "deployment"]

    # --- Production evidence (plan §E) ---
    PRODUCTION_KEYWORDS = ["production", "deployed", "deploy", "scaled", "scale", "monitoring",
                           "latency", "pipeline", "pipelines", "users", "serving", "real-time",
                           "real time", "throughput", "uptime", "sla"]

    # --- Research evidence (plan §6) ---
    RESEARCH_KEYWORDS = ["research", "phd", "publication", "published", "paper", "papers",
                         "academic", "thesis", "novel", "state-of-the-art", "arxiv"]

    # --- Off-domain (penalty G) ---
    CV_KEYWORDS       = ["yolo", "opencv", "segmentation", "object detection", "cnn",
                         "image classification", "computer vision", "gans", "diffusion models"]
    SPEECH_KEYWORDS   = ["asr", "wav2vec", "audio", "tts", "speech recognition", "speech"]
    ROBOTICS_KEYWORDS = ["ros", "slam", "motion planning", "robotics", "lidar"]

    # --- Like-to-have bonus groups (plan §F) ---
    LIKE_TO_HAVE = {
        "fine_tuning":     ["lora", "qlora", "peft", "fine-tun"],
        "learning_to_rank":["lambdamart", "xgboost ranker", "learning to rank", "neural rank"],
        "hr_tech":         ["recruitment", "talent matching", "hiring", "hr-tech", "hr tech",
                            "recruiting", "marketplace"],
        "distributed":     ["inference optimization", "distributed serving", "distributed systems",
                            "model serving", "triton"],
        "open_source":     ["open source", "open-source", "github", "contributor", "maintainer"],
    }

    # --- Consulting / services firms (plan §4) ---
    CONSULTING_FIRMS = ["tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
                        "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "mphasis",
                        "deloitte", "pwc", "kpmg", "ernst & young", "ey ", "igate", "syntel",
                        "hexaware", "birlasoft", "persistent systems"]

    # --- Non-tech job categories (plan §5) — entire-career-non-tech filter ---
    NON_TECH_TITLES = ["sales", "marketing", "human resources", "hr manager", "recruiter",
                       "recruitment", "talent acquisition", "finance", "accountant", "accounting",
                       "operations manager", "customer success", "customer support", "legal",
                       "business development", "account manager", "project manager", "product manager",
                       "content writer", "designer", "mechanical", "civil engineer", "supply chain"]
    TECH_WHITELIST  = ["data analyst", "analytics engineer", "data engineer", "bi engineer",
                       "research engineer", "solutions architect", "software engineer",
                       "ml engineer", "machine learning", "ai engineer", "data scientist",
                       "backend engineer", "platform engineer", "devops", "developer"]

    print("Lexicons loaded:",
          f"{len(AIML_SKILLS)} AI/ML skills,",
          f"{len(RELEVANT_TITLES)} title patterns,",
          f"{len(CONSULTING_FIRMS)} consulting firms.")
    # ===== cell e5b9bc8b =====
    # --------------------------- helpers --------------------------------------
    def _open_any(path):
        return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") \
            else open(path, "r", encoding="utf-8")

    def _lower(x):
        return str(x).lower() if x is not None else ""

    def any_kw(text, keywords):
        return any(k in text for k in keywords)

    def count_kw(text, keywords):
        return sum(1 for k in keywords if k in text)

    def _is_tech_job(title_l, desc_l):
        if any(k in title_l for k in TECH_WHITELIST):
            return True
        if any(k in title_l for k in RELEVANT_TITLES):
            return True
        if any_kw(desc_l, CAREER_EVIDENCE):
            return True
        return False

    def _is_aiml_job(title_l, desc_l):
        aiml_titles = ["ml engineer", "machine learning", "ai engineer", "applied scientist",
                       "data scientist", "nlp", "research scientist", "research engineer",
                       "search engineer", "relevance", "recommendation", "ranking", "applied ai",
                       "deep learning", "computer vision", "mlops"]
        return any(k in title_l for k in aiml_titles) or any_kw(desc_l, IR_CORE_KEYWORDS) \
            or any_kw(desc_l, ["machine learning", "deep learning", "model", "nlp", "embedding"])

    # --------------------------- single-pass load -----------------------------
    t0 = time.time()
    records = []
    with _open_any(DATA_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            prof = c.get("profile", {}) or {}
            sig  = c.get("redrob_signals", {}) or {}
            skills = c.get("skills", []) or []
            career = c.get("career_history", []) or []

            # ---- text aggregation (lowercased once) ----
            headline = _lower(prof.get("headline"))
            summary  = _lower(prof.get("summary"))
            cur_title = _lower(prof.get("current_title"))
            skill_names = [_lower(s.get("name")) for s in skills]
            skills_text = " ".join(skill_names)
            titles_all  = " | ".join([cur_title] + [_lower(e.get("title")) for e in career])
            career_text = " ".join(_lower(e.get("description")) for e in career)
            full_text   = " ".join([headline, summary, cur_title, skills_text, titles_all, career_text])

            # ---- skills features ----
            n_expert = sum(1 for s in skills if _lower(s.get("proficiency")) == "expert")
            expert_zero_dur = sum(1 for s in skills
                                  if _lower(s.get("proficiency")) == "expert"
                                  and (s.get("duration_months") or 0) == 0)
            has_relevant_skill = any(sn in AIML_SKILLS for sn in skill_names) or \
                                 any(any(a in sn for a in AIML_SKILLS) for sn in skill_names)

            # ---- title / career evidence ----
            title_relevant  = any(t in titles_all for t in RELEVANT_TITLES) or \
                              _is_aiml_job(cur_title, "")
            career_evidence = any_kw(career_text + " " + summary, CAREER_EVIDENCE)

            # ---- tenure / hopping / domain ratios ----
            total_jobs = len(career)
            companies  = {_lower(e.get("company")) for e in career}
            total_companies = len(companies)
            durations  = [int(e.get("duration_months") or 0) for e in career]
            total_months = sum(durations)
            short_count = sum(1 for d in durations if 0 < d < 18)
            short_switch_ratio = short_count / total_jobs if total_jobs else 0.0

            consulting_months = sum(int(e.get("duration_months") or 0) for e in career
                                    if any(cf in _lower(e.get("company")) for cf in CONSULTING_FIRMS))
            consulting_ratio = consulting_months / total_months if total_months else 0.0

            # research vs production
            research_months = sum(int(e.get("duration_months") or 0) for e in career
                                  if any_kw(_lower(e.get("title")) + " " + _lower(e.get("description")),
                                            RESEARCH_KEYWORDS))
            research_ratio = research_months / total_months if total_months else 0.0
            production_evidence = any_kw(career_text + " " + summary, PRODUCTION_KEYWORDS)

            # all-non-tech?  (no single job qualifies as tech)
            non_tech_all = total_jobs > 0 and not any(
                _is_tech_job(_lower(e.get("title")), _lower(e.get("description"))) for e in career)

            # applied AI/ML years = tenure in jobs that look like AI/ML work,
            # capped at total years_of_experience (cannot exceed the career length)
            applied_ai_months = sum(int(e.get("duration_months") or 0) for e in career
                                    if _is_aiml_job(_lower(e.get("title")), _lower(e.get("description"))))
            applied_ai_years = applied_ai_months / 12.0
            _yoe = prof.get("years_of_experience")
            if isinstance(_yoe, (int, float)):
                applied_ai_years = min(applied_ai_years, float(_yoe))

            # ---- career trajectory (promotion trend + tenure stability) ----
            # Order roles chronologically by start_date, then read the seniority sequence.
            _roles = [(e.get("start_date") or "", seniority_rank(_lower(e.get("title"))))
                      for e in career]
            _roles.sort(key=lambda x: x[0])            # ascending start_date; undated roles first
            _sen_seq = [r[1] for r in _roles]
            career_delta = (_sen_seq[-1] - _sen_seq[0]) if len(_sen_seq) >= 2 else 0
            current_seniority = seniority_rank(cur_title) if cur_title else (
                _sen_seq[-1] if _sen_seq else 3)
            avg_tenure_months = (total_months / total_jobs) if total_jobs else 0.0

            records.append({
                "candidate_id": c.get("candidate_id"),
                "headline": headline, "summary": summary, "location": _lower(prof.get("location")),
                "country": _lower(prof.get("country")), "current_title": cur_title,
                "years_of_experience": prof.get("years_of_experience"),
                "current_company": _lower(prof.get("current_company")),
                "current_industry": _lower(prof.get("current_industry")),
                "skills_text": skills_text, "titles_all": titles_all,
                "career_text": career_text, "full_text": full_text,
                "has_relevant_skill": has_relevant_skill, "title_relevant": title_relevant,
                "career_evidence": career_evidence, "n_expert": n_expert,
                "expert_zero_dur": expert_zero_dur, "total_months": total_months,
                "total_jobs": total_jobs, "total_companies": total_companies,
                "short_switch_ratio": short_switch_ratio, "consulting_ratio": consulting_ratio,
                "non_tech_all": non_tech_all, "research_ratio": research_ratio,
                "production_evidence": production_evidence, "applied_ai_years": applied_ai_years,
                # career-trajectory features
                "career_delta": career_delta, "current_seniority": current_seniority,
                "avg_tenure_months": avg_tenure_months,
                # behavioral signals
                "open_to_work": bool(sig.get("open_to_work_flag", False)),
                "willing_to_relocate": bool(sig.get("willing_to_relocate", False)),
                "last_active_date": sig.get("last_active_date"),
                "recruiter_response_rate": sig.get("recruiter_response_rate"),
                "interview_completion_rate": sig.get("interview_completion_rate"),
                "notice_period_days": sig.get("notice_period_days"),
                "github_activity_score": sig.get("github_activity_score"),
                "profile_completeness_score": sig.get("profile_completeness_score"),
            })

    feat = pd.DataFrame(records).set_index("candidate_id")
    del records

    # recency: days since last active, relative to the most-recent activity in the pool
    feat["last_active_date"] = pd.to_datetime(feat["last_active_date"], errors="coerce")
    REF_DATE = feat["last_active_date"].max()
    if pd.isna(REF_DATE):
        REF_DATE = pd.Timestamp(datetime.utcnow().date())
    feat["days_since_active"] = (REF_DATE - feat["last_active_date"]).dt.days
    feat["years_of_experience"] = pd.to_numeric(feat["years_of_experience"], errors="coerce")

    print(f"Loaded & featurized {len(feat):,} candidates in {time.time()-t0:.1f}s "
          f"({feat.shape[1]} features). Recency reference date: {REF_DATE.date()}")
    # ===== cell 0c3f1533 =====
    # --------------------------- Stage 1 filters ------------------------------
    def in_preferred_location(loc):
        return any(h in loc for h in PREFERRED_LOCATIONS)

    elim = pd.DataFrame(index=feat.index)

    # 1. Location
    elim["loc_reject"] = (
        ~feat["location"].apply(in_preferred_location) & ~feat["willing_to_relocate"]
    )
    # 2. No AI/ML evidence (skill OR title OR career)
    elim["no_ai_evidence"] = ~(
        feat["has_relevant_skill"] | feat["title_relevant"] | feat["career_evidence"]
    )
    # 3. Extreme job-hopping (only when >=3 companies)
    elim["job_hopper"] = (feat["total_companies"] >= 3) & (feat["short_switch_ratio"] > 0.7)
    # 4. Consulting-only career
    elim["consulting_only"] = feat["consulting_ratio"] >= 0.999
    # 5. Entire career non-tech
    elim["non_tech"] = feat["non_tech_all"]
    # 6. Pure research without production
    elim["pure_research"] = (feat["research_ratio"] >= 0.999) & (~feat["production_evidence"])

    elim["eliminated"] = elim.drop(columns=[]).any(axis=1)

    # ------------------------------ funnel log --------------------------------
    print("=== STAGE 1: DIRECT ELIMINATIONS ===\n")
    print(f"{'Total pool':37s}: {len(feat):>7,}")
    # report marginal (unique-cause) and gross counts for transparency
    for col, label in [
        ("loc_reject",      "Rejected: location"),
        ("no_ai_evidence",  "Rejected: no AI/ML evidence"),
        ("job_hopper",      "Rejected: extreme job-hopping"),
        ("consulting_only", "Rejected: consulting-only"),
        ("non_tech",        "Rejected: entire-career non-tech"),
        ("pure_research",   "Rejected: pure research (no prod)"),
    ]:
        print(f"  {label:35s}: {int(elim[col].sum()):>7,}")

    survivors = feat[~elim["eliminated"]].copy()
    elim_survivors = elim.loc[survivors.index]  # (all False) kept for parity
    print(f"\n{'Total eliminated (any rule)':37s}: {int(elim['eliminated'].sum()):>7,}")
    print(f"{'SURVIVORS -> Stage 2':37s}: {len(survivors):>7,}  "
          f"({len(survivors)/len(feat)*100:.1f}% of pool)")
    # ===== cell cf03a4e6 =====
    ts = time.time()

    # =================== A. LOCATION SCORE ====================================
    def _loc_score(loc, country, relo):
        if any(h in loc for h in PREFERRED_PRIMARY):   return 1.00
        if any(h in loc for h in PREFERRED_SECONDARY): return 0.95
        if "india" in country:                         return 0.80 if relo else 0.60
        return 0.40 if relo else 0.20                  # outside India
    survivors["score_location"] = [
        _loc_score(l, c, r) for l, c, r in
        zip(survivors["location"], survivors["country"], survivors["willing_to_relocate"])
    ]

    # =================== B. BEHAVIORAL SCORE ==================================
    def _notice_score(d):
        if d is None or (isinstance(d, float) and np.isnan(d)): return 0.5
        if d <= 30:  return 1.00
        if d <= 60:  return 0.85
        if d <= 90:  return 0.60
        if d <= 120: return 0.30
        return 0.10
    def _recency_score(d):                         # exponential decay (~120-day scale)
        if d is None or (isinstance(d, float) and np.isnan(d)): return 0.4
        return float(np.clip(np.exp(-d / 120.0), 0.0, 1.0))
    def _num(x, default):
        return float(x) if x is not None and not pd.isna(x) else default

    beh = []
    for otw, dsa, rr, ic, npd in zip(
            survivors["open_to_work"], survivors["days_since_active"],
            survivors["recruiter_response_rate"], survivors["interview_completion_rate"],
            survivors["notice_period_days"]):
        s = (0.28 * (1.0 if otw else 0.0)      # open-to-work  (very high priority)
             + 0.27 * _recency_score(dsa)       # last active   (very high priority)
             + 0.18 * _num(rr, 0.5)             # recruiter response rate
             + 0.12 * _num(ic, 0.5)             # interview completion rate
             + 0.15 * _notice_score(npd))       # notice period (piecewise)
        beh.append(s)
    survivors["score_behavioral"] = beh

    # =================== C. EXPERIENCE SCORE ==================================
    def _applied_ai_score(y):                      # peak preference 4-5y
        if y < 1:   return 0.00
        if y < 2:   return 0.20
        if y < 3:   return 0.40
        if y < 4:   return 0.70
        if y <= 5:  return 1.00
        if y <= 7:  return 0.90
        if y <= 10: return 0.75
        return 0.55
    def _total_exp_score(y):                       # peak preference 5-9y
        if y is None or np.isnan(y): return 0.30
        if y < 3:   return 0.00
        if y < 5:   return 0.60
        if y <= 9:  return 1.00
        if y <= 15: return 0.80
        return 0.60

    # --- Career-trajectory signal (promotion trend + tenure stability) ---
    # Rewards climbing the seniority ladder, reaching senior/lead altitude, and healthy
    # tenure (not job-hopping). Feeds the experience score as a third component so the
    # rubric weights stay unchanged (experience already covers "career quality").
    def _trajectory_score(delta, cur_rank, avg_tenure, short_ratio):
        prog = float(np.clip(0.5 + 0.25 * delta, 0.0, 1.0))            # climbed the ladder?
        alt  = {1: 0.10, 2: 0.30, 3: 0.50, 4: 0.80, 5: 0.95, 6: 1.00}.get(int(cur_rank), 0.50)
        if   avg_tenure <= 0:    stab = 0.5                            # unknown tenure -> neutral
        elif avg_tenure < 12:    stab = 0.3                            # churny
        elif avg_tenure <= 72:   stab = 1.0                            # healthy 1-6y stints
        else:                    stab = 0.8                            # very long single stint
        stab *= (1.0 - 0.5 * min(float(short_ratio), 1.0))            # job-hopping haircut
        return 0.40 * prog + 0.35 * alt + 0.25 * stab

    survivors["score_trajectory"] = [
        _trajectory_score(d, cs, at, ss)
        for d, cs, at, ss in zip(survivors["career_delta"], survivors["current_seniority"],
                                 survivors["avg_tenure_months"], survivors["short_switch_ratio"])
    ]
    survivors["score_experience"] = [
        0.5 * _applied_ai_score(a) + 0.3 * _total_exp_score(t) + 0.2 * tr
        for a, t, tr in zip(survivors["applied_ai_years"], survivors["years_of_experience"],
                            survivors["score_trajectory"])
    ]

    # =================== E. PRODUCTION EVIDENCE ===============================
    survivors["score_production"] = [
        float(np.clip(count_kw(ct + " " + sm, PRODUCTION_KEYWORDS) / 4.0, 0.0, 1.0))
        for ct, sm in zip(survivors["career_text"], survivors["summary"])
    ]

    # =================== F. LIKE-TO-HAVE BONUS ===============================
    _lth_groups = list(LIKE_TO_HAVE.values())
    survivors["score_like_to_have"] = [
        sum(1 for kws in _lth_groups if any_kw(ft, kws)) / len(_lth_groups)
        for ft in survivors["full_text"]
    ]

    # =================== G. CV/SPEECH/ROBOTICS PENALTY =======================
    def _penalty(ft):
        ir  = count_kw(ft, IR_CORE_KEYWORDS)
        off = max(count_kw(ft, CV_KEYWORDS), count_kw(ft, SPEECH_KEYWORDS), count_kw(ft, ROBOTICS_KEYWORDS))
        if ir == 0 and off >= 2: return 0.65, "off-domain (CV/Speech/Robotics) focus, no IR evidence"
        if ir == 0 and off == 1: return 0.85, "limited IR evidence, some off-domain focus"
        return 1.0, ""
    _pen = [_penalty(ft) for ft in survivors["full_text"]]
    survivors["penalty_mult"]   = [p[0] for p in _pen]
    survivors["penalty_reason"] = [p[1] for p in _pen]

    # =================== H. RELOCATION-RISK PENALTY ==========================
    # Outside-India candidates (score_location <= 0.40) face high relocation/visa friction
    # for a Pune/Noida role. Apply a multiplicative haircut so they survive ONLY if
    # technically exceptional. Keyed on score_location (not raw country) so hub candidates
    # with a missing country field are never mislabelled. Compounds with the off-domain penalty.
    RELOCATION_PENALTY = 0.85
    _abroad = survivors["score_location"] < 0.5
    survivors.loc[_abroad, "penalty_mult"] = survivors.loc[_abroad, "penalty_mult"] * RELOCATION_PENALTY
    survivors.loc[_abroad, "penalty_reason"] = survivors.loc[_abroad, "penalty_reason"].apply(
        lambda r: (r + "; " if r else "") + "outside-India relocation/visa risk")
    print(f"Relocation penalty ({RELOCATION_PENALTY}) applied to {int(_abroad.sum()):,} outside-India survivors.")

    print(f"Sub-scores A/B/C(+trajectory)/E/F/G computed for {len(survivors):,} survivors in {time.time()-ts:.1f}s")
    # ===== cell daec518d =====
    # ============== JD REPRESENTATION (parsed from job_description.md) ==============
    import os

    def _load_jd_markdown():
        for p in ("jd/job_description_crux.md", "jd/job_description.md",
                  "organizer/job_description_crux.md", "organizer/job_description.md",
                  "job_description_crux.md", "job_description.md",
                  "./jd/job_description_crux.md", "./jd/job_description.md",
                  "./organizer/job_description_crux.md", "./organizer/job_description.md"):
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
        return ""

    def _extract_section_bullets(md, header):
        """Bullets under a '## header' until the next markdown header line."""
        out, capture = [], False
        for ln in md.splitlines():
            s = ln.strip()
            if s.startswith("#"):
                capture = header.lower() in s.lower()
                continue
            if capture and s.startswith("-"):
                txt = re.sub(r"\*\*(.*?)\*\*", r"\1", s.lstrip("-").strip())
                out.append(re.sub(r"[*_`]", "", txt))
        return out

    _md         = _load_jd_markdown()
    must_have   = _extract_section_bullets(_md, "Things you absolutely need")
    like_have   = _extract_section_bullets(_md, "Things we'd like you to have")
    do_not_want = _extract_section_bullets(_md, "Things we explicitly do NOT want")

    # Curated technical seed (tool/vocab grounding) — retains the prior JD_QUERY signal.
    TECH_SEED = (
        "embeddings based retrieval ranking search recommendation semantic search information retrieval "
        "vector database pinecone weaviate qdrant milvus faiss elasticsearch opensearch learning to rank "
        "bm25 re-ranking relevance fine-tuning large language models llm rag transformers sentence embeddings "
        "hugging face evaluation ndcg mrr map precision recall offline online a/b testing production "
        "low-latency scalable machine learning pipelines serving mlops model serving monitoring feature engineering python"
    )

    # Robust fallbacks if the markdown can't be parsed (clean-sandbox safety).
    if not must_have:
        must_have = ["embeddings-based retrieval systems deployed to production",
                     "vector databases and hybrid search infrastructure", "strong python",
                     "evaluation frameworks for ranking ndcg mrr map offline online a/b testing"]
    if not like_have:
        like_have = ["llm fine-tuning lora qlora peft", "learning-to-rank xgboost neural",
                     "hr-tech recruiting marketplace", "distributed systems inference optimization",
                     "open-source ai ml contributions"]
    if not do_not_want:
        do_not_want = ["title-chasers switching companies every 1.5 years",
                       "framework enthusiasts langchain tutorials demos",
                       "only worked at consulting firms tcs infosys wipro accenture cognizant capgemini",
                       "primary expertise computer vision speech or robotics without nlp ir",
                       "entirely closed-source proprietary work without external validation"]

    # Must-have repeated 2x => ~2x weight in BM25/TF-IDF + dense-centroid shift.
    # Order matters: must-haves first so the encoder's token truncation drops like_have, not must_have.
    JD_QUERY = " . ".join([TECH_SEED, " . ".join(must_have), " . ".join(must_have),
                           " . ".join(like_have)]).lower()

    # Negative representation: do-not-want bullets + explicit off-domain vocabulary.
    JD_NEGATIVE = (" . ".join(do_not_want) + " . " +
                   "computer vision opencv yolo object detection segmentation cnn image classification . "
                   "speech asr wav2vec tts audio . robotics ros slam motion planning lidar . "
                   "consulting services outsourcing . pure academic research only no production").lower()

    # Natural-language query for the cross-encoder (MS-MARCO style; NOT keyword-stuffed).
    JD_CE_QUERY = ("Senior AI engineer with production experience building embeddings-based retrieval, "
                   "ranking, and semantic search systems using vector databases, with LLM fine-tuning and "
                   "ranking evaluation such as NDCG and MRR, deployed to real users at scale.")

    print(f"Parsed JD -> must_have:{len(must_have)} like_have:{len(like_have)} do_not_want:{len(do_not_want)}")

    _token_re = re.compile(r"[a-z0-9#+.]+")
    def tokenize(text):
        return _token_re.findall(text)

    def _minmax(arr):
        arr = np.asarray(arr, dtype=float)
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

    surv_ids = survivors.index.tolist()
    docs = survivors["full_text"].tolist()
    components = {}
    ts = time.time()

    # --- (1) TF-IDF cosine ---
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    min_df = 1 if len(docs) < 5 else 2
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=50000,
                            sublinear_tf=True, stop_words="english")
    X = tfidf.fit_transform(docs + [JD_QUERY])
    tfidf_sim = cosine_similarity(X[-1], X[:-1]).ravel()
    components["tfidf"] = _minmax(tfidf_sim)
    print(f"TF-IDF cosine computed for {len(docs):,} docs.")

    # --- (2) BM25 ---
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([tokenize(d) for d in docs])
        components["bm25"] = _minmax(bm25.get_scores(tokenize(JD_QUERY)))
        print("BM25 computed.")
    except Exception as e:
        print("BM25 unavailable:", e)

    # --- (3) keyword evidence (IR + eval + LLM cues), normalized ---
    kw_ev = np.array([
        (count_kw(ft, IR_CORE_KEYWORDS) + count_kw(ft, EVAL_KEYWORDS) + count_kw(ft, LLM_KEYWORDS))
        for ft in survivors["full_text"]
    ], dtype=float)
    survivors["jd_keyword_evidence"] = _minmax(kw_ev)

    # --- combine available lexical components ---
    lex_weights = {"bm25": 0.45, "tfidf": 0.30, "keyword": 0.25}
    present = {k: components[k] for k in ("bm25", "tfidf") if k in components}
    wsum = sum(lex_weights[k] for k in present) + lex_weights["keyword"]
    jd_align = (lex_weights["keyword"] / wsum) * survivors["jd_keyword_evidence"].values
    for k, v in present.items():
        jd_align = jd_align + (lex_weights[k] / wsum) * v
    survivors["score_jd_alignment"] = jd_align

    print(f"\nPhase-1 JD alignment computed for ALL {len(survivors):,} survivors "
          f"in {time.time()-ts:.1f}s. Backends: {['keyword'] + list(present.keys())}")
    print(survivors["score_jd_alignment"].describe().round(4).to_string())

    # ============== RELEVANCE-FIRST embed_text (truncation-aware, model input) =====
    # full_text stays untouched (used by BM25/TF-IDF/keyword counting, which never truncate).
    # embed_text packs the most JD-relevant, de-duplicated signal first so it survives the
    # encoder's token cap (bge = 512).  Order: JD-relevant skills -> title -> summary -> titles -> headline -> career.
    EMBED_MAX_WORDS = 400
    _EMBED_PRIORITY = set(IR_CORE_KEYWORDS) | set(EVAL_KEYWORDS) | set(LLM_KEYWORDS) | set(AIML_SKILLS)

    def _dedupe_cap(text, max_words=EMBED_MAX_WORDS):
        seen, out = set(), []
        for w in text.split():
            if w in seen:
                continue
            seen.add(w); out.append(w)
            if len(out) >= max_words:
                break
        return " ".join(out)

    def _build_embed_text(sk, ct, sm, ta, hl, cr):
        skills = sk.split()
        rel = [s for s in skills if any(p in s for p in _EMBED_PRIORITY)]
        rel_set = set(rel)
        oth = [s for s in skills if s not in rel_set]
        parts = [" ".join(rel + oth), ct, sm, ta.replace("|", " "), hl, cr]
        return _dedupe_cap(" ".join(p for p in parts if p))

    survivors["embed_text"] = [
        _build_embed_text(sk, ct, sm, ta, hl, cr)
        for sk, ct, sm, ta, hl, cr in zip(
            survivors["skills_text"], survivors["current_title"], survivors["summary"],
            survivors["titles_all"], survivors["headline"], survivors["career_text"])
    ]
    print(f"Built relevance-first embed_text (<= {EMBED_MAX_WORDS} words) for {len(survivors):,} survivors.")
    # ===== cell c077ac7a =====
    # ---- Preliminary weighted aggregation (Phase 1 — no dense embeddings yet) ----
    def aggregate(df):
        base = sum(WEIGHTS[c] * df[f"score_{c}"] for c in WEIGHTS)
        return base, base * df["penalty_mult"]

    survivors["base_score"], survivors["final_score"] = aggregate(survivors)
    print(f"Preliminary (Phase-1) final score computed for {len(survivors):,} candidates.")
    print(survivors[["score_jd_alignment", "score_experience", "score_behavioral",
                     "score_production", "score_location", "score_like_to_have",
                     "penalty_mult", "final_score"]].describe().round(3).to_string())
    # ===== cell adaptive_shortlist_k =====
    # Adaptive shortlist size (K) by elbow + percentile guard
    # Because dense embeddings are precomputed offline, K can be data-driven rather than throughput-limited.

    scores = survivors["final_score"].sort_values(ascending=False).values
    n = len(scores)
    if n <= 1:
        K = max(SHORTLIST_MIN, min(n, SHORTLIST_MAX))
        reason = "pool too small"
    else:
        # Kneedle: largest perpendicular distance from line (first -> last)
        x = np.arange(n, dtype=np.float64)
        y = scores.astype(np.float64)
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
        a = y_norm[-1] - y_norm[0]
        b = x_norm[0] - x_norm[-1]
        c = x_norm[-1] * y_norm[0] - x_norm[0] * y_norm[-1]
        denom = np.sqrt(a * a + b * b) + 1e-12
        dist = np.abs(a * x_norm + b * y_norm + c) / denom
        elbow_idx = int(np.argmax(dist)) + 1  # K = number of points before the elbow

        # Percentile guard: everyone at/above the 95th percentile of score
        pct_score = np.percentile(scores, 95)
        pct_idx = int(np.sum(scores >= pct_score))

        K_raw = max(elbow_idx, pct_idx)
        K = int(np.clip(K_raw, SHORTLIST_MIN, SHORTLIST_MAX))
        reason = f"elbow={elbow_idx}, percentile={pct_idx}"

    print(f"Adaptive shortlist size K = {K:,} ({reason}, min={SHORTLIST_MIN}, max={SHORTLIST_MAX})")

    # ===== cell b328a18f =====
    # ===== Offline embedding cache + FAISS index over the K contender set =====
    # This cell runs in the 5-minute ranking window. It assumes the full-pool embedding cache
    # (bge_embeddings.npz) was already built in the pre-step. It simply loads the cache and builds
    # an exact-cosine FAISS index over the shortlisted contenders for fast dense retrieval.

    import hashlib, os

    contender_ids = survivors.nlargest(K, "final_score").index.tolist()

    def _txt_hash(t):
        return hashlib.md5(t.encode("utf-8")).hexdigest()

    # --- load the full-pool cache (candidate_id -> (text_hash, vector)) ---
    _full_cache = {}
    if os.path.exists(EMB_CACHE_PATH):
        _z = np.load(EMB_CACHE_PATH, allow_pickle=True)
        _full_cache = {str(cid): (str(h), v) for cid, h, v in zip(_z["ids"], _z["hashes"], _z["vecs"])}

    # --- verify the shortlist contenders are present in the cache ---
    _missing = [cid for cid in contender_ids if str(cid) not in _full_cache]
    if _missing:
        print(f"WARNING: {len(_missing):,} / {len(contender_ids):,} shortlisted contenders missing from cache. "
              f"Falling back to lexical-only for those ids; run the pre-step embedding cell to avoid this.")
        contender_ids = [cid for cid in contender_ids if str(cid) in _full_cache]
        K = len(contender_ids)

    # --- build the contender-only FAISS index from the cached vectors ---
    surv_emb = None
    faiss_index = None
    if USE_DENSE_EMBEDDINGS and contender_ids:
        try:
            surv_emb = np.vstack([_full_cache[str(c)][1] for c in contender_ids]).astype(np.float32)
            print(f"Loaded {len(contender_ids):,} contender embeddings from cache "
                  f"({len(_full_cache):,} total pool vectors cached).")
            if USE_FAISS and surv_emb is not None:
                import faiss
                faiss_index = faiss.IndexFlatIP(surv_emb.shape[1])
                faiss_index.add(surv_emb)
                print(f"FAISS IndexFlatIP built over {surv_emb.shape[0]:,} contender vectors.")
        except Exception as e:
            print(f"Dense index build failed ({e}); dense retrieval will be skipped.")
            USE_DENSE_EMBEDDINGS = False
    else:
        print("No cached contenders available; dense retrieval will be skipped.")
    # ===== cell 75160bde =====
    # Contender set = the K rows whose embeddings were precomputed + FAISS-indexed above.
    shortlist_ids = contender_ids
    surv_pos = {cid: i for i, cid in enumerate(surv_ids)}

    dense_applied = False
    if USE_DENSE_EMBEDDINGS and surv_emb is not None:
        ts = time.time()
        try:
            st_model = globals().get("_dense_model")
            if st_model is None:
                from sentence_transformers import SentenceTransformer
                st_model = SentenceTransformer(EMBED_MODEL)
                globals()["_dense_model"] = st_model
            st_model.max_seq_length = EMBED_MAX_TOKENS
            # bge: the QUERY (JD) carries the retrieval instruction; passages (cached) do not.
            q_emb   = st_model.encode([BGE_QUERY_INSTRUCTION + JD_QUERY],
                                      normalize_embeddings=True).astype(np.float32)
            neg_emb = st_model.encode([BGE_QUERY_INSTRUCTION + JD_NEGATIVE],
                                      normalize_embeddings=True).astype(np.float32)
            NEG_PENALTY = 0.5  # how strongly off-domain / do-not-want profiles are pushed down

            # positive relevance via FAISS ANN (exact inner-product); fall back to a dense matmul.
            if faiss_index is not None:
                D, I = faiss_index.search(q_emb, surv_emb.shape[0])
                pos = np.empty(surv_emb.shape[0], dtype=float); pos[I[0]] = D[0]
            else:
                pos = (surv_emb @ q_emb.T).ravel()
            neg = (surv_emb @ neg_emb.T).ravel()
            dense_signal = pos - NEG_PENALTY * neg
            dense_norm = _minmax(dense_signal)

            # pull lexical components for the contenders (same min-max space as Phase 1)
            sl_bm25  = _minmax(np.array([components.get("bm25", np.zeros(len(surv_ids)))[surv_pos[c]]
                                         for c in shortlist_ids]))
            sl_tfidf = _minmax(np.array([components.get("tfidf", np.zeros(len(surv_ids)))[surv_pos[c]]
                                         for c in shortlist_ids]))
            sl_kw    = survivors.loc[shortlist_ids, "jd_keyword_evidence"].values

            # refined hybrid JD alignment: dense-dominant, lexical-supported
            jd_v2 = 0.50 * dense_norm + 0.25 * sl_bm25 + 0.10 * sl_tfidf + 0.15 * sl_kw
            survivors.loc[shortlist_ids, "score_jd_alignment"] = jd_v2

            # recompute base + final over the full contender set (vectorized over the slice)
            sl = survivors.loc[shortlist_ids]
            b, fin = aggregate(sl)
            survivors.loc[shortlist_ids, "base_score"]  = b
            survivors.loc[shortlist_ids, "final_score"] = fin
            dense_applied = True
            _eng = "FAISS" if faiss_index is not None else "NumPy dot"
            print(f"Dense (bge + {_eng}) retrieval applied to {len(shortlist_ids):,} contenders "
                  f"in {time.time()-ts:.1f}s (positive - {NEG_PENALTY}*negative).")
        except Exception as e:
            print(f"Dense retrieval unavailable ({e}); keeping Phase-1 lexical scores.")

    if not dense_applied:
        print("Dense step skipped — ranking on BM25 + TF-IDF + keyword evidence only.")

    # ===== Stage 2E — Cross-encoder re-rank (precise, top-CE_TOPK only) ==========
    # A cross-encoder scores each (JD, resume) pair jointly -> sharper relevance than the
    # bi-encoder cosine, but it is O(pairs) so we run it only on the strongest head.
    ce_applied = False
    if USE_CROSS_ENCODER and dense_applied:
        ts = time.time()
        try:
            ce_model = globals().get("_ce_model")
            if ce_model is None:
                from sentence_transformers import CrossEncoder
                ce_model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=EMBED_MAX_TOKENS)
                globals()["_ce_model"] = ce_model
            ce_ids = survivors.loc[shortlist_ids].nlargest(
                min(CE_TOPK, len(shortlist_ids)), "final_score").index.tolist()
            pairs = [[JD_CE_QUERY, survivors.at[c, "embed_text"]] for c in ce_ids]
            ce_raw = ce_model.predict(pairs, batch_size=32, show_progress_bar=True)
            ce_norm = _minmax(np.asarray(ce_raw, dtype=float))

            # blend the cross-encoder signal into JD alignment for the re-ranked head.
            # Cross-encoder-dominant: it is the most reliable relevance signal; the bi-encoder
            # term is kept only as a stabilizing prior.
            CE_BLEND = 0.75
            cur = survivors.loc[ce_ids, "score_jd_alignment"].values
            survivors.loc[ce_ids, "score_jd_alignment"] = CE_BLEND * ce_norm + (1 - CE_BLEND) * cur

            sl2 = survivors.loc[ce_ids]
            b2, fin2 = aggregate(sl2)
            survivors.loc[ce_ids, "base_score"]  = b2
            survivors.loc[ce_ids, "final_score"] = fin2
            ce_applied = True
            print(f"Cross-encoder re-ranked top {len(ce_ids):,} in {time.time()-ts:.1f}s "
                  f"({CE_BLEND:.2f}*cross + {1-CE_BLEND:.2f}*bi-encoder).")
        except Exception as e:
            print(f"Cross-encoder unavailable ({e}); keeping bi-encoder ranking.")

    if not ce_applied:
        print("Cross-encoder step skipped — keeping bi-encoder ranking.")

    if len(shortlist_ids) > 0:
        print(f"\nFinal-score stats over contender set (K={K}):")
        print(survivors.loc[shortlist_ids, "final_score"].describe().round(4).to_string())
    else:
        print(f"\nFinal-score stats over contender set (K={K}): empty shortlist")
    # ===== cell cf816790 =====
    HONEYPOT_REJECT_THRESHOLD = 3   # cumulative suspicion points -> hard reject
    SENIOR_TITLE_TOKENS = ["senior", "lead", "staff", "principal", "head", "chief", "director"]

    def honeypot_flags(cid, r):
        flags, pts = [], 0
        # H1: expert skills with ~zero duration
        ezd = r["expert_zero_dur"]
        if ezd >= 3:   flags.append(f"{ezd} expert skills with ~0 months usage"); pts += 3
        elif ezd == 2: flags.append("2 expert skills with ~0 months usage");      pts += 2
        # H2: implausible expert breadth
        if r["n_expert"] >= 10: flags.append(f"{r['n_expert']} expert skills (implausible breadth)"); pts += 2
        # H3: stated experience vs career tenure
        yoe = r["years_of_experience"]; career_years = r["total_months"] / 12.0
        if yoe is not None and not np.isnan(yoe) and career_years > 0:
            gap = abs(yoe - career_years)
            if gap > 8:   flags.append(f"exp/tenure mismatch ({yoe:.0f}y stated vs {career_years:.0f}y career)"); pts += 3
            elif gap > 5: flags.append(f"exp/tenure gap (~{gap:.0f}y)"); pts += 1
        # H4: title inflation
        if any(k in r["current_title"] for k in SENIOR_TITLE_TOKENS) and career_years < 2:
            flags.append("senior-level title with <2y total tenure"); pts += 2
        return pts, "; ".join(flags)

    _hp = [honeypot_flags(cid, r) for cid, r in survivors.iterrows()]
    survivors["honeypot_points"] = [h[0] for h in _hp]
    survivors["honeypot_reason"] = [h[1] for h in _hp]
    survivors["is_honeypot"]     = survivors["honeypot_points"] >= HONEYPOT_REJECT_THRESHOLD

    # soft penalty for uncertain cases (1-2 points)
    soft = survivors["honeypot_points"].between(1, HONEYPOT_REJECT_THRESHOLD - 1)
    survivors.loc[soft, "final_score"] *= 0.90

    clean = survivors[~survivors["is_honeypot"]].copy()
    print("=== STAGE 3: HONEYPOT DETECTION ===")
    print(f"  Hard-rejected (>= {HONEYPOT_REJECT_THRESHOLD} pts): {int(survivors['is_honeypot'].sum()):,}")
    print(f"  Soft-penalized (1-2 pts)     : {int(soft.sum()):,}")
    print(f"  Clean candidates remaining   : {len(clean):,}")
    # ===== cell 6712ac4b =====
    # ----------------------------- Select Top 100 -----------------------------
    # Sort by score desc; break ties by candidate_id ASC (spec-compliant + deterministic).
    clean = clean.reset_index()
    clean = clean.sort_values(["final_score", "candidate_id"], ascending=[False, True])
    top = clean.head(TOP_N).copy().reset_index(drop=True)
    top["rank"] = range(1, len(top) + 1)

    if len(top) == 0:
        print("WARNING: no candidates left after honeypot filtering; output will be empty.")
    else:
        hp_in_top = int(top["is_honeypot"].sum())
        print(f"Top {len(top)} selected. Honeypots in shortlist: {hp_in_top} "
              f"({hp_in_top/len(top)*100:.1f}%) — must stay < 10%.")
        assert hp_in_top / len(top) < 0.10, "Honeypot rate >= 10% in top 100!"
    # ===== cell ac1e6d29 =====
    # --------------------------- Concise, data-driven reasoning generation -----
    # Inspired by the friend's approach: each reasoning is built from actual candidate
    # facts (title, years, skills, career highlight, behavioral signals, concerns)
    # rather than rotated templates. Fast, deterministic, and offline-safe.

    def _matching_skills(r, top_n=3):
        """Return the most relevant JD-matching skills from the candidate's skill set."""
        text = " " + r["skills_text"] + " "
        hits = [skill for skill in AIML_SKILLS if skill in text]
        # Drop substring duplicates (e.g. keep "llms", drop "llm")
        hits = [h for h in hits if not any(h != other and h in other for other in hits)]
        # Prefer retrieval / core AI skills, then longer/more specific names
        core = {"information retrieval", "semantic search", "vector search", "recommendation systems",
                "ranking systems", "learning to rank", "bm25", "faiss", "embeddings", "sentence transformers",
                "nlp", "natural language processing", "machine learning", "deep learning", "pytorch"}
        hits.sort(key=lambda s: (0 if s in core else 1, -len(s)))
        return hits[:top_n]


    def _career_highlight(r):
        """Pick one concrete career highlight from the candidate's history."""
        career_text = r["career_text"]
        titles = r["titles_all"]
        company = r["current_company"] if r["current_company"] else "product org"
        context = (career_text + " " + titles).lower()

        if any_kw(career_text, ["production", "deployed", "serving", "inference", "real-time", "real time"]):
            return f"production ML experience at {company}"
        if any_kw(context, ["ranking", "search", "retrieval", "recommendation", "relevance", "semantic search"]):
            return f"search/ranking/retrieval experience at {company}"
        if any_kw(career_text, ["embeddings", "vector", "faiss", "bm25", "ndcg", "mrr", "learning to rank"]):
            return f"embedding/retrieval tooling experience at {company}"
        if company and not any(cf in company for cf in CONSULTING_FIRMS):
            return f"currently at {company}"
        return None


    def _behavioral_note(r):
        """Short availability / engagement note."""
        parts = []
        if r["open_to_work"]:
            parts.append("open to work")
        rr = r["recruiter_response_rate"]
        if rr is not None and not pd.isna(rr) and rr >= 0.7:
            parts.append(f"response rate {rr:.0%}")
        dsa = r["days_since_active"]
        if dsa is not None and not pd.isna(dsa) and dsa <= 30:
            parts.append("recently active")
        if parts:
            return ", ".join(parts)
        return None


    def _concerns(r):
        """Return a list of honest, data-backed concerns (empty if none)."""
        concerns = []
        npd = r["notice_period_days"]
        if npd is not None and not pd.isna(npd) and npd > 60:
            concerns.append(f"notice period {int(npd)}d")
        if r["score_location"] < 0.5:
            concerns.append("location/visa risk")
        avg_tenure = r["avg_tenure_months"]
        if avg_tenure is not None and not pd.isna(avg_tenure) and 0 < avg_tenure < 12:
            concerns.append(f"short avg tenure ({avg_tenure:.0f}mo)")
        rr = r["recruiter_response_rate"]
        if rr is not None and not pd.isna(rr) and rr < 0.3:
            concerns.append(f"low response rate ({rr:.0%})")
        if r["penalty_reason"]:
            concerns.append(r["penalty_reason"])
        if r["honeypot_reason"]:
            concerns.append(f"minor flag: {r['honeypot_reason']}")
        return concerns


    def _strongest_score_axis(r):
        """Return the single strongest scoring contributor for the opening phrase."""
        contrib = {
            "JD alignment": WEIGHTS["jd_alignment"] * r["score_jd_alignment"],
            "experience":   WEIGHTS["experience"]   * r["score_experience"],
            "behavioral":   WEIGHTS["behavioral"]   * r["score_behavioral"],
            "production":   WEIGHTS["production"]   * r["score_production"],
            "location":     WEIGHTS["location"]     * r["score_location"],
        }
        return max(contrib, key=contrib.get)


    def make_reasoning(r):
        """Build a concise 1-2 sentence justification from real candidate facts."""
        parts = []

        # 1) Title + years of experience
        title = r["current_title"].title() if r["current_title"] else "Candidate"
        yoe = r["years_of_experience"]
        aai = r["applied_ai_years"]
        if yoe is not None and not pd.isna(yoe) and yoe > 0:
            exp_part = f"{title} with {yoe:.1f}y total experience"
            if aai > 0:
                exp_part += f" ({aai:.1f}y applied AI/ML)"
            parts.append(exp_part)
        else:
            parts.append(title)

        # 2) Top matching skills
        skills = _matching_skills(r)
        if skills:
            parts.append(" + ".join(skills))

        # 3) Career highlight
        highlight = _career_highlight(r)
        if highlight:
            parts.append(highlight)

        # 4) Behavioral signals
        behav = _behavioral_note(r)
        if behav:
            parts.append(behav)

        # 5) Notice period (always mention if known)
        npd = r["notice_period_days"]
        if npd is not None and not pd.isna(npd) and npd > 0:
            parts.append(f"notice period {int(npd)}d")

        # Build main sentence
        reasoning = "; ".join(parts) + "."

        # 6) Concerns, if any
        concerns = _concerns(r)
        if concerns:
            reasoning += " Concern: " + "; ".join(concerns) + "."
        else:
            reasoning += " No major concerns."

        # 6b) Skill-gap note only when there are other concerns and the full profile lacks
        # explicit evaluation/ranking evidence (JD must-have signals).
        if concerns:
            profile_text = " " + str(r["full_text"]).lower() + " "
            if not any(kw in profile_text for kw in ("evaluation framework", "ndcg", "mrr", "map", "ab testing")):
                reasoning += " Skill note: no explicit evaluation/ranking metrics (NDCG/MRR/MAP/A-B) in profile."

        # 7) Tie to the JD's top scoring axis (short, non-hallucinatory)
        strongest = _strongest_score_axis(r)
        if strongest != "JD alignment":
            reasoning += f" Strongest signal: {strongest}."

        return re.sub(r"\s+", " ", reasoning).strip()


    top["reasoning"] = [make_reasoning(r) for _, r in top.iterrows()]

    if len(top) > 0:
        for _, r in top.head(5).iterrows():
            print(f"[{int(r['rank']):>3}] {r['candidate_id']}  score={r['final_score']:.4f}")
            print("     ", r["reasoning"], "\n")
    # ===== cell f61924d2 =====
    # ----------------------------- Write submission ---------------------------
    submission = top[["candidate_id", "rank", "final_score", "reasoning"]].rename(
        columns={"final_score": "score"})
    submission["score"] = submission["score"].round(6)
    submission = submission.sort_values("rank").reset_index(drop=True)

    # safety: scores must be non-increasing by rank (already true after the sort/tie-break)
    assert (submission["score"].diff().dropna() <= 1e-9).all(), "scores must be non-increasing"
    assert len(submission) <= TOP_N, f"expected at most {TOP_N} rows, got {len(submission)}"
    assert submission["candidate_id"].is_unique and submission["rank"].is_unique
    if len(submission) < TOP_N:
        print(f"WARNING: sample produced only {len(submission)} rows (expected {TOP_N}). "
              "This is acceptable for sandbox demos but the full submission must use the full dataset.")

    submission.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote {OUTPUT_CSV} with {len(submission)} rows.")
    print(f"Total ranking time: {time.time()-t0:.1f}s (target < 300s)\n")

    # ----------------------- Validate with the OFFICIAL validator --------------
    import importlib.util, os
    if os.path.exists("validator/validate_submission.py"):
        _vp = "validator/validate_submission.py"
    elif os.path.exists("organizer/validate_submission.py"):
        _vp = "organizer/validate_submission.py"
    else:
        _vp = "validate_submission.py"
    _spec = importlib.util.spec_from_file_location("validate_submission", _vp)
    _vmod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_vmod)
    if len(submission) == TOP_N:
        _errors = _vmod.validate_submission(OUTPUT_CSV)
        if _errors:
            print(f"VALIDATION FAILED ({len(_errors)} issue(s)):")
            for e in _errors:
                print(" -", e)
        else:
            print("VALIDATION PASSED — submission.csv conforms to the spec.")
    else:
        print("Skipping official validator: output is not a full 100-row submission.")
    # ===== cell 7a37691c =====
    # --------------------------- QA / sanity summary --------------------------
    print("=== TOP 100 QA SUMMARY ===")
    if len(top) > 0:
        print(f"Score range        : {top['final_score'].min():.4f} -> {top['final_score'].max():.4f}")
        print(f"Honeypots in top   : {int(top['is_honeypot'].sum())} (DQ if > 10)")
        print(f"Unique reasonings  : {top['reasoning'].nunique()} / {len(top)}")
    else:
        print("No candidates in the top shortlist.")

    if len(top) > 0:
        loc_mix = top["location"].apply(
            lambda l: next((h for h in PREFERRED_LOCATIONS if h in l), "other/relocate")).value_counts()
        print("\nLocation mix (top 100):"); print(loc_mix.to_string())

        # ---- Relocation diagnostic: what is actually inside 'other/relocate'? ----
        # Every non-hub survivor is willing_to_relocate (Stage-1 guarantee), so they split into:
        #   score_location == 0.80  -> India, willing to relocate (low risk, JD-acceptable)
        #   score_location == 0.40  -> outside India, willing to relocate (visa/relocation risk)
        _other = top[~top["location"].apply(in_preferred_location)].copy()
        in_hub = len(top) - len(_other)
        india_relo  = int((_other["score_location"] >= 0.79).sum())   # 0.80 tier
        abroad_relo = int((_other["score_location"] <  0.79).sum())   # 0.40 tier
        print("\n'other/relocate' composition (top 100):")
        print(f"  In preferred hub (0.95-1.00)      : {in_hub}")
        print(f"  India + willing to relocate (0.80): {india_relo}")
        print(f"  Outside India + willing  (0.40)   : {abroad_relo}   <- relocation/visa risk")
        if abroad_relo:
            print("\n  Outside-India candidates in top 100 (id | country | rank | score):")
            for _, rr in _other[_other["score_location"] < 0.79].sort_values("rank").iterrows():
                print(f"    {rr['candidate_id']} | {str(rr['country']).title():15s} | "
                      f"#{int(rr['rank']):3d} | {rr['final_score']:.4f}")

        print("\nYears of experience (top 100):")
        print(top["years_of_experience"].describe().round(1).to_string())
        print("\nApplied-AI years (top 100):")
        print(top["applied_ai_years"].describe().round(1).to_string())
        print("\nMean sub-scores (top 100):")
        print(top[[f"score_{c}" for c in WEIGHTS]].mean().round(3).to_string())
        submission.head(10)


def main():
    parser = argparse.ArgumentParser(description="Rank top-100 candidates for the Redrob hackathon.")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates.jsonl(.gz)")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    args = parser.parse_args()
    run_pipeline(args.candidates, args.out)
    print(f"Submission written to {args.out}")


if __name__ == "__main__":
    main()
