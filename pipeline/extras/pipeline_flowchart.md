# Candidate Ranking Pipeline — Flowchart

Recruiter-style ranking of the **Top 100** candidates for the *Founding Senior AI Engineer* role.

**Guiding philosophy:** *Eliminate only candidates who are extremely unlikely to be selected per the JD. Everything else is expressed as preferences through scoring.*

Dense BGE embeddings are built **offline once for the entire candidate pool** via `get_embeddings.py` (download from HuggingFace if possible, otherwise generate locally) and persisted to `pipeline/cache/bge_embeddings_completed.npz`. This is the **precomputation step** and must complete before the ranking step starts.

At ranking time `rank.py` only loads the prebuilt full-pool cache and builds a **FAISS index over the shortlisted contenders**, so Phase 2 retrieval covers **all K contenders** while staying inside the 5-minute CPU budget. `rank.py` does **not** download or generate embeddings. Cheap lexical signals still do the heavy filtering first.

This `pipeline/` folder is self-contained: it holds the code, data, cache, job descriptions, validator, and generated output.

---

## High-level flow

```mermaid
flowchart TD
    A["100,000 Candidates<br/>data/candidates.jsonl"] --> B

    subgraph S0["Stage 0 — Load &amp; Feature Engineering"]
        B["Single vectorized pass<br/>lowercase + aggregate text<br/>+ career trajectory features:<br/>career_delta, current_seniority,<br/>avg_tenure_months, short_switch_ratio"]
    end

    B --> P0["Pre-step (offline)<br/>BGE embeddings for full pool<br/>get_embeddings.py<br/>download or generate<br/>cache/bge_embeddings_completed.npz<br/>id + text-hash keyed"]

    P0 --> C

    subgraph S1["Stage 1 — Direct Eliminations (hard rejects)"]
        C{"Reject if ANY rule fires"}
        C --> C1["Location: not preferred hub<br/>AND not willing to relocate"]
        C --> C2["No AI/ML evidence<br/>(skill / title / career)"]
        C --> C3["Extreme job-hopping<br/>≥3 companies &amp; short-ratio &gt; 0.7"]
        C --> C4["Consulting-only career<br/>consulting_ratio == 1"]
        C --> C5["Entire-career non-tech"]
        C --> C6["Pure research, no production"]
    end

    C1 & C2 & C3 & C4 & C5 & C6 --> D["Survivors ≈ 26K<br/>(~26% of pool)"]

    D --> JD["JD preprocessing<br/>parse markdown bullets:<br/>must-have, like-have, do-not-want<br/>jd/job_description_crux.md"]
    JD --> JDQ["JD_QUERY = TECH_SEED + must_have×2 + like_have<br/>JD_NEGATIVE = do_not_want + off-domain terms<br/>JD_CE_QUERY = natural-language CE prompt"]
    JDQ --> RES["Resume preprocessing<br/>skills_text + current_title + summary<br/>+ titles_all + headline + career_text<br/>dedupe, JD-relevant skills first<br/>≤400 words / ≤512 tokens"]
    RES --> E

    subgraph S2["Stage 2 — Candidate Scoring (all survivors)"]
        E["Six sub-scores in [0,1]"]
        E --> E1["A. Location (0.0548)"]
        E --> E2["B. Behavioral (0.1644)<br/>0.28·open_to_work + 0.27·recency<br/>+ 0.18·response + 0.12·interview<br/>+ 0.15·notice_period"]
        E --> E3["C. Experience (0.2192)<br/>0.5·applied AI + 0.3·total exp<br/>+ 0.2·career trajectory"]
        E3 --> E3T["Career trajectory<br/>promotion trend + tenure stability<br/>anti job-hopping"]
        E --> E4["D. JD Alignment (0.3424)<br/>Phase-1: 0.45·BM25 + 0.30·TF-IDF<br/>+ 0.25·keyword evidence"]
        E --> E5["E. Production (0.1370)<br/>count(production keywords) / 4"]
        E --> E6["F. Like-to-have (0.0822)<br/>fine-tuning, LTR, HR-tech,<br/>distributed systems, open-source"]
        E --> E7["G. Off-domain penalty ×<br/>+ relocation penalty (0.85)"]
    end

    E1 & E2 & E3 & E4 & E5 & E6 & E7 --> F["base = Σ WEIGHTS·score<br/>final = base × penalty_mult<br/>(Phase-1 preliminary score)"]

    F --> G["Adaptive shortlist size K<br/>(elbow detection + percentile guard)<br/>no live-encode budget clamp"]

    G --> H

    subgraph S2D["Phase 2 — BGE + FAISS + Cross-encoder (contender set)"]
        H["Load full-pool cache<br/>cache/bge_embeddings_completed.npz<br/>id + text-hash keyed"]
        H --> H0["Filter to K shortlisted<br/>contender vectors"]
        H0 --> H1["FAISS IndexFlatIP<br/>384-dim BAAI/bge-small-en-v1.5"]
        H1 --> I["BGE query: JD_QUERY<br/>ANN dense = pos − 0.5·neg vector"]
        I --> I1["Refine JD alignment:<br/>0.50·dense + 0.25·BM25<br/>+ 0.10·TF-IDF + 0.15·kw"]
        I1 --> J["Cross-encoder re-rank top 400<br/>cross-encoder/ms-marco-MiniLM-L-6-v2<br/>blend 0.75·CE + 0.25·bi"]
        J --> K["Recompute final_score<br/>for contender set"]
    end

    K --> L

    subgraph S3["Stage 3 — Honeypot Detection"]
        L{"Accumulate suspicion points"}
        L --> L1["Expert skills w/ ~0 months"]
        L --> L2["Implausible expert breadth"]
        L --> L3["Stated exp vs tenure mismatch"]
        L --> L4["Title inflation (&lt;2y tenure)"]
        L1 & L2 & L3 & L4 --> M{"points ≥ 3?"}
        M -->|"Yes"| M1["Hard reject"]
        M -->|"1-2 pts"| M2["Soft penalize ×0.90"]
        M -->|"0"| M3["Clean"]
    end

    M1 -.->|"removed"| N
    M2 --> N
    M3 --> N["Sort by final_score desc<br/>tie-break: candidate_id asc"]

    N --> O["Select Top 100<br/>assert honeypot rate &lt; 10%"]

    O --> P["Reasoning generation<br/>(title + yrs, skills, career highlight,<br/>behavioral signals, data-backed concerns,<br/>skill note only when concerns exist +<br/>full profile lacks eval/ranking metrics)"]

    P --> Q["output/submission.csv<br/>candidate_id, rank, score, reasoning"]

    Q --> R{"validator/validate_submission.py"}
    R -->|"PASS"| S["Final submission ✓"]
```

---

## Scoring weights (JD-grounded rubric)

`weight ∝ decisiveness × discriminativeness`, normalised to sum to exactly 1.0.

| Component | Decisiveness | Discriminativeness | Weight |
| --- | --- | --- | --- |
| **JD alignment** | 5 | 5 | 0.3424 |
| **Experience** | 4 | 4 | 0.2192 |
| **Behavioral** | 3 | 4 | 0.1644 |
| **Production** | 5 | 2 | 0.1370 |
| **Like-to-have** | 2 | 3 | 0.0822 |
| **Location** | 2 | 2 | 0.0548 |

---

## What is not considered

**Protected / sensitive attributes:** gender, age, date of birth, ethnicity, religion, marital status, disability, salary, expected salary range, candidate photo, anonymized name.

**Unused structured profile fields:** `education` (degree, institution, field of study, grade, tier), `certifications`, `languages`, `current_company_size`, career-history `is_current` / `industry` / `company_size`, skill `endorsements`.

**Unused Redrob signals:** `profile_completeness_score`, `signup_date`, `profile_views_received_30d`, `applications_submitted_30d`, `avg_response_time_hours`, `skill_assessment_scores`, `connection_count`, `endorsements_received`, `expected_salary_range_inr_lpa`, `preferred_work_mode`, `github_activity_score`, `search_appearance_30d`, `saved_by_recruiters_30d`, `offer_acceptance_rate`, `verified_email`, `verified_phone`, `linkedin_connected`.

**Other:** no external API, LLM, or human-in-the-loop is used during ranking; all decisions are deterministic.

---

## Compute constraints

- **≤ 5 min** wall-clock for the ranking step, **CPU only**, **≤ 16 GB RAM**, **no network during ranking**.
- Model download / `pip install` / offline embedding cache build happen in the **pre-step script** (`get_embeddings.py`) allowed outside the 5-min window.
- Dense BGE embeddings are persisted to `cache/bge_embeddings_completed.npz` and indexed with FAISS, so the ranking step loads them instantly.
- Output: exactly **100 rows**, `score` monotonically non-increasing, ties broken by `candidate_id` ascending.
- **Honeypot rate &gt; 10% in the top 100 = disqualification.**
