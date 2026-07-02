"""Extract and print complete details of honeypot candidates for manual analysis."""
import json, gzip, re, sys
import numpy as np
import pandas as pd

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

# Lexicons (reused from notebook)
PREFERRED_LOCATIONS = ["pune", "noida", "delhi", "ncr", "gurgaon", "gurugram",
                       "mumbai", "navi mumbai", "hyderabad"]
AIML_SKILLS = {
    "information retrieval", "information retrieval systems", "semantic search", "vector search",
    "recommendation systems", "ranking systems", "learning to rank", "bm25", "faiss", "pinecone",
    "weaviate", "milvus", "qdrant", "pgvector", "elasticsearch", "opensearch", "haystack",
    "search backend", "search infrastructure", "search & discovery", "indexing algorithms",
    "content matching", "vector representations", "text encoders",
    "embeddings", "sentence transformers", "hugging face transformers", "llms", "llm", "rag",
    "langchain", "llamaindex", "prompt engineering", "fine-tuning llms", "lora", "qlora", "peft",
    "nlp", "natural language processing", "model adaptation",
    "machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn", "data science",
    "feature engineering", "statistical modeling", "reinforcement learning", "time series",
    "forecasting", "model adaptation",
    "mlops", "mlflow", "kubeflow", "bentoml", "weights & biases", "open-source ml libraries",
}
RELEVANT_TITLES = [
    "ml engineer", "machine learning engineer", "ai engineer", "applied scientist",
    "applied ai", "data scientist", "research engineer", "nlp engineer", "search engineer",
    "relevance engineer", "recommendation engineer", "ranking engineer", "mlops engineer",
    "ml scientist", "research scientist", "deep learning", "computer vision engineer",
    "data engineer", "analytics engineer", "data analyst", "bi engineer",
    "solutions architect", "software engineer", "backend engineer", "platform engineer",
]
CAREER_EVIDENCE = [
    "retrieval", "ranking", "recommendation", "relevance", "search", "embedding", "embeddings",
    "semantic", "vector", "nlp", "language model", "recommender", "personalization",
    "matching", "evaluation", "deployed ml", "ml system", "ml pipeline", "machine learning",
]
IR_CORE_KEYWORDS = [
    "retrieval", "ranking", "recommendation", "relevance", "search", "embedding", "embeddings",
    "semantic search", "vector", "bm25", "faiss", "learning to rank", "recommender", "matching",
]
EVAL_KEYWORDS = ["ndcg", "mrr", "map", "evaluation", "offline", "online", "benchmark",
                 "a/b", "ab test", "ab testing", "precision", "recall"]
LLM_KEYWORDS = ["prompt engineering", "fine-tun", "lora", "qlora", "peft", "llm", "rag",
                "large language model", "transformer", "deployment"]
PRODUCTION_KEYWORDS = ["production", "deployed", "deploy", "scaled", "scale", "monitoring",
                       "latency", "pipeline", "pipelines", "users", "serving", "real-time",
                       "real time", "throughput", "uptime", "sla"]
RESEARCH_KEYWORDS = ["research", "phd", "publication", "published", "paper", "papers",
                     "academic", "thesis", "novel", "state-of-the-art", "arxiv"]
CV_KEYWORDS       = ["yolo", "opencv", "segmentation", "object detection", "cnn",
                     "image classification", "computer vision", "gans", "diffusion models"]
SPEECH_KEYWORDS   = ["asr", "wav2vec", "audio", "tts", "speech recognition", "speech"]
ROBOTICS_KEYWORDS = ["ros", "slam", "motion planning", "robotics", "lidar"]
LIKE_TO_HAVE = {
    "fine_tuning":     ["lora", "qlora", "peft", "fine-tun"],
    "learning_to_rank":["lambdamart", "xgboost ranker", "learning to rank", "neural rank"],
    "hr_tech":         ["recruitment", "talent matching", "hiring", "hr-tech", "hr tech",
                        "recruiting", "marketplace"],
    "distributed":     ["inference optimization", "distributed serving", "distributed systems",
                        "model serving", "triton"],
    "open_source":     ["open source", "open-source", "github", "contributor", "maintainer"],
}
CONSULTING_FIRMS = ["tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
                    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "mphasis",
                    "deloitte", "pwc", "kpmg", "ernst & young", "ey ", "igate", "syntel",
                    "hexaware", "birlasoft", "persistent systems"]
TECH_WHITELIST  = ["data analyst", "analytics engineer", "data engineer", "bi engineer",
                   "research engineer", "solutions architect", "software engineer",
                   "ml engineer", "machine learning", "ai engineer", "data scientist",
                   "backend engineer", "platform engineer", "devops", "developer"]

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

# --------------------------- load & feature engineering -------------------
DATA_PATH = "candidates.jsonl"
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

        headline = _lower(prof.get("headline"))
        summary  = _lower(prof.get("summary"))
        cur_title = _lower(prof.get("current_title"))
        skill_names = [_lower(s.get("name")) for s in skills]
        skills_text = " ".join(skill_names)
        titles_all  = " | ".join([cur_title] + [_lower(e.get("title")) for e in career])
        career_text = " ".join(_lower(e.get("description")) for e in career)
        full_text   = " ".join([headline, summary, cur_title, skills_text, titles_all, career_text])

        n_expert = sum(1 for s in skills if _lower(s.get("proficiency")) == "expert")
        expert_zero_dur = sum(1 for s in skills
                              if _lower(s.get("proficiency")) == "expert"
                              and (s.get("duration_months") or 0) == 0)
        has_relevant_skill = any(sn in AIML_SKILLS for sn in skill_names) or \
                             any(any(a in sn for a in AIML_SKILLS) for sn in skill_names)

        title_relevant  = any(t in titles_all for t in RELEVANT_TITLES) or \
                          _is_aiml_job(cur_title, "")
        career_evidence = any_kw(career_text + " " + summary, CAREER_EVIDENCE)

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

        research_months = sum(int(e.get("duration_months") or 0) for e in career
                              if any_kw(_lower(e.get("title")) + " " + _lower(e.get("description")),
                                        RESEARCH_KEYWORDS))
        research_ratio = research_months / total_months if total_months else 0.0
        production_evidence = any_kw(career_text + " " + summary, PRODUCTION_KEYWORDS)

        non_tech_all = total_jobs > 0 and not any(
            _is_tech_job(_lower(e.get("title")), _lower(e.get("description"))) for e in career)

        applied_ai_months = sum(int(e.get("duration_months") or 0) for e in career
                                if _is_aiml_job(_lower(e.get("title")), _lower(e.get("description"))))
        applied_ai_years = applied_ai_months / 12.0
        _yoe = prof.get("years_of_experience")
        if isinstance(_yoe, (int, float)):
            applied_ai_years = min(applied_ai_years, float(_yoe))

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

# --------------------------- honeypot detection ---------------------------
HONEYPOT_REJECT_THRESHOLD = 3
SENIOR_TITLE_TOKENS = ["senior", "lead", "staff", "principal", "head", "chief", "director"]

def honeypot_flags(cid, r):
    flags, pts = [], 0
    ezd = r["expert_zero_dur"]
    if ezd >= 3:   flags.append(f"{ezd} expert skills with ~0 months usage"); pts += 3
    elif ezd == 2: flags.append("2 expert skills with ~0 months usage");      pts += 2
    if r["n_expert"] >= 10: flags.append(f"{r['n_expert']} expert skills (implausible breadth)"); pts += 2
    yoe = r["years_of_experience"]; career_years = r["total_months"] / 12.0
    if yoe is not None and not np.isnan(yoe) and career_years > 0:
        gap = abs(yoe - career_years)
        if gap > 8:   flags.append(f"exp/tenure mismatch ({yoe:.0f}y stated vs {career_years:.0f}y career)"); pts += 3
        elif gap > 5: flags.append(f"exp/tenure gap (~{gap:.0f}y)"); pts += 1
    if any(k in r["current_title"] for k in SENIOR_TITLE_TOKENS) and career_years < 2:
        flags.append("senior-level title with <2y total tenure"); pts += 2
    return pts, "; ".join(flags)

_hp = [honeypot_flags(cid, r) for cid, r in feat.iterrows()]
feat["honeypot_points"] = [h[0] for h in _hp]
feat["honeypot_reason"] = [h[1] for h in _hp]
feat["is_honeypot"]     = feat["honeypot_points"] >= HONEYPOT_REJECT_THRESHOLD

# --------------------------- print honeypot details -----------------------
hard = feat[feat["is_honeypot"]].copy()
soft = feat[feat["honeypot_points"].between(1, HONEYPOT_REJECT_THRESHOLD - 1)].copy()

print("=" * 80)
print(f"HONEYPOT ANALYSIS — {len(hard)} hard-rejected (>= {HONEYPOT_REJECT_THRESHOLD} pts), "
      f"{len(soft)} soft-penalized (1-2 pts)")
print("=" * 80)

def print_group(df, label):
    if df.empty:
        print(f"\n{label}: NONE")
        return
    print(f"\n{label} ({len(df)} candidates):")
    print("-" * 80)
    for cid, r in df.iterrows():
        print(f"\nCANDIDATE: {cid}")
        print(f"  Points: {r['honeypot_points']} | Reason: {r['honeypot_reason'] or 'N/A'}")
        print(f"  Title: {r['current_title']}")
        print(f"  Company: {r['current_company']}")
        print(f"  Location: {r['location']}")
        print(f"  Years of experience: {r['years_of_experience']}")
        print(f"  Applied-AI years: {r['applied_ai_years']:.1f}")
        print(f"  Total months (career): {r['total_months']}")
        print(f"  Total jobs: {r['total_jobs']} | Total companies: {r['total_companies']}")
        print(f"  Expert skills: {r['n_expert']} | Expert zero-duration: {r['expert_zero_dur']}")
        print(f"  Short-switch ratio: {r['short_switch_ratio']:.2f}")
        print(f"  Consulting ratio: {r['consulting_ratio']:.2f}")
        print(f"  Research ratio: {r['research_ratio']:.2f}")
        print(f"  Production evidence: {r['production_evidence']}")
        print(f"  Open to work: {r['open_to_work']} | Willing to relocate: {r['willing_to_relocate']}")
        print(f"  Headline: {r['headline'][:120]}...")
        print(f"  Summary: {r['summary'][:120]}...")
        print(f"  Skills: {r['skills_text'][:120]}...")

print_group(hard, "HARD-REJECTED HONEYPOTS")
print_group(soft, "SOFT-PENALIZED (SUSPICIOUS)")

print("\n" + "=" * 80)
print("EXPORTING TO honeypots.csv for easy analysis...")
cols = ["honeypot_points", "honeypot_reason", "current_title", "current_company", "location",
        "years_of_experience", "applied_ai_years", "total_months", "total_jobs", "total_companies",
        "n_expert", "expert_zero_dur", "short_switch_ratio", "consulting_ratio", "research_ratio",
        "production_evidence", "open_to_work", "willing_to_relocate", "headline", "summary", "skills_text"]
all_hp = pd.concat([hard, soft])
all_hp[cols].to_csv("honeypots.csv", index=True, index_label="candidate_id")
print(f"Saved {len(all_hp)} honeypot candidates to honeypots.csv")
