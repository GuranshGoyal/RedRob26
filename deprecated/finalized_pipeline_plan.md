# Redrob Hackathon – Finalized Candidate Ranking Pipeline (Implementation Specification)

## Objective

Build a recruiter-style AI ranking system that identifies the Top 100 candidates most suitable for the given Founding Senior AI Engineer role.

The system should progressively narrow down the candidate pool by combining:

* Hard eliminations
* Recruiter-inspired scoring
* Semantic matching
* Behavioral signals
* Honeypot detection
* Explainable reasoning

The guiding philosophy is:

> «Eliminate only those candidates who are extremely unlikely to be selected according to the JD. Everything else should be expressed as preferences through scoring.»

---

# Overall Pipeline

```
100,000 Candidates
        │
        ▼
Stage 1: Direct Eliminations
        │
        ▼
Filtered Candidate Pool
        │
        ▼
Stage 2: Candidate Scoring
        │
        ▼
Top Candidates
        │
        ▼
Stage 3: Honeypot Detection
        │
        ▼
Final Top 100
        │
        ▼
Reasoning Generation
        │
        ▼
submission.csv
```

---

# Stage 1: Direct Eliminations

Candidates satisfying ANY of the following conditions are eliminated.

---

## 1. Location Filter

Reject candidate if:

```
(location NOT IN {
    Pune,
    Noida,
    Delhi NCR,
    Mumbai,
    Hyderabad
})
AND
(willing_to_relocate == False)
```

### Redrob Signal Used:

* willing_to_relocate

---

## 2. AI/ML Evidence Filter

Reject candidate if NONE of the following are true.

### Skill Evidence

At least one relevant skill exists.

Relevant skill set consists of:

* AI-generated exhaustive mapping derived from:

  * 133 unique skills in dataset
  * JD terminology
  * relevant synonyms

AND

```
(if experience duration exists)
skill_experience > 0
```

---

### Title Evidence

Current title is relevant.

OR

Past title is relevant.

Examples:

* ML Engineer
* Data Scientist
* Search Engineer
* Relevance Engineer
* Recommendation Engineer
* NLP Engineer
* Applied Scientist
* AI Engineer

etc.

---

### Career Description Evidence

Career descriptions contain evidence of relevant work.

Examples:

* retrieval
* ranking
* recommendation
* relevance
* search
* embeddings
* evaluation
* deployed ML systems

---

Reject if ALL above fail.

---

## 3. Extreme Job-Hopping Filter

Only applicable if:

```
total_companies ≥ 3
```

Compute:

```
short_switch_ratio =
companies_with_tenure_<18_months / total_companies
```

Reject if:

```
short_switch_ratio > 0.7
```

---

## 4. Consulting-only Career Filter

Create consulting company dictionary using:

* JD companies
* Dataset EDA

Compute:

```
consulting_ratio =
consulting_months / total_months
```

Reject if:

```
consulting_ratio == 1
```

---

## 5. Entire Career Non-Tech Filter

Reject if ALL jobs belong to non-tech categories.

Examples:

* HR
* Sales
* Marketing
* Finance
* Operations
* Customer Success
* Legal
* Recruitment

Maintain whitelist for:

* Data Analyst
* Analytics Engineer
* Data Engineer
* BI Engineer
* Research Engineer
* Solutions Architect

etc.

---

## 6. Pure Research Filter (Optional)

Reject only if:

```
research_ratio == 1
AND
production_evidence == 0
```

Production evidence examples:

* deployed
* production
* users
* latency
* monitoring
* pipelines
* scaled

---

# Stage 2: Candidate Scoring

All surviving candidates receive scores.

Final score is obtained using weighted aggregation.

---

## A. Location Score

| Condition                      | Score |
| ------------------------------ | ----- |
| Pune / Noida                   | 1.00  |
| Delhi NCR / Mumbai / Hyderabad | 0.95  |
| India + willing to relocate    | 0.80  |
| India + relocation unknown     | 0.60  |
| Outside India + willing        | 0.40  |

---

## B. Behavioral Score

### Redrob Signals Used:

### Open To Work

Very High Priority.

Binary bonus.

---

### Last Active Date

Very High Priority.

Lower recency → higher score.

---

### Recruiter Response Rate

High Priority.

Higher is better.

---

### Interview Completion Rate

High Priority.

Higher is better.

---

### Notice Period

Piecewise scoring.

| Notice Period | Score |
| ------------- | ----- |
| ≤30 days      | 1.00  |
| 31–60         | 0.85  |
| 61–90         | 0.60  |
| 91–120        | 0.30  |
| >120          | 0.10  |

---

## C. Experience Score

Two independent components.

---

### Applied AI/ML Experience

Peak preference.

| Applied AI Years | Score |
| ---------------- | ----- |
| <1               | 0.00  |
| 2                | 0.40  |
| 3                | 0.70  |
| 4–5              | 1.00  |
| 6–7              | 0.90  |
| 8–10             | 0.75  |
| >10              | 0.55  |

---

### Total Experience

Peak preference.

| Total Years | Score |
| ----------- | ----- |
| <3          | 0.00  |
| 3–4         | 0.60  |
| 5–9         | 1.00  |
| 10–15       | 0.80  |
| >15         | 0.60  |

---

## D. JD Alignment Score

Use hybrid retrieval.

### Representation:

```
Current Title
+ Past Titles
+ Career Descriptions
+ Skills
```

Compare against positive JD representation.

---

### Retrieval Signals

* retrieval
* search
* relevance
* ranking
* recommendation
* matching

---

### Evaluation Signals

* ndcg
* mrr
* map
* evaluation
* offline
* online
* benchmark
* A/B testing

---

### LLM Integration Signals

Evidence of:

* prompt engineering
* fine tuning
* deployment
* LLM systems

Do NOT infer opinions.

Only infer evidence.

---

### Hybrid similarity:

```
BM25
+ Dense Embeddings
+ Cosine Similarity
```

---

## E. Production Evidence Score

High Priority.

Keywords:

* production
* deployed
* scaled
* monitoring
* latency
* pipelines
* users
* serving
* real-time

Higher evidence → higher score.

---

## F. Like-to-Have Bonus Score

Small bonus component.

Examples:

### Fine-Tuning

* LoRA
* QLoRA
* PEFT

---

### Learning-to-Rank

* LambdaMART
* XGBoost Ranker
* Neural Rankers

---

### HR-Tech Exposure

* recruitment
* talent matching
* hiring systems

---

### Distributed Systems

* inference optimization
* distributed serving

---

### Open Source

AI/ML contributions.

---

## G. CV / Robotics / Speech Penalty

Compute:

### IR Score

Keywords:

* retrieval
* ranking
* recommendation
* search
* relevance
* evaluation

---

### CV Score

* yolo
* opencv
* segmentation
* object detection
* cnn

---

### Speech Score

* asr
* wav2vec
* audio
* tts

---

### Robotics Score

* ros
* slam
* motion planning

---

Apply penalty if:

```
IR_score ≈ 0
AND
(CV_score OR Speech_score OR Robotics_score high)
```

No penalty if meaningful IR evidence exists.

---

# Final Weighted Aggregation

Final candidate score:

```
Weighted Average(
    Location Score,
    Behavioral Score,
    Experience Score,
    JD Alignment Score,
    Production Evidence Score,
    Like-to-Have Bonus,
    CV/Robotics Penalty
)
```

Weights to be finalized through pairwise recruiter-style comparisons derived from the JD.

---

# Stage 3: Honeypot Detection

Direct elimination.

Reject if:

* Impossible timelines.
* Contradictory roles.
* Expert skill with zero duration.
* Impossible career progression.
* Strong evidence of title inflation.
* Logically inconsistent profiles.

If uncertainty exists:

Use penalty instead of rejection.

---

# Final Selection

```
Sort candidates by Final Score ↓
Select Top 100
```

---

# Reasoning Generation

Generate explanations ONLY for Top 100.

Reasoning should be derived from the ranking decision.

Include:

* Top 2–3 strongest contributors
* One limitation or concern
* Connection to JD requirements

Avoid:

* Hallucinations
* Generic statements
* Repeating the same explanation structure

---

# Deliverables

## 1. submission.csv

### Required Fields:

* candidate_id
* rank
* score
* reasoning

## 2. Reproducible ranking pipeline.

## 3. Hosted sandbox/demo environment.

## 4. GitHub repository with complete implementation.
