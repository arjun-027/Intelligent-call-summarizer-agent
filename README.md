# Insurance Call Summariser Agent

**Upload a call transcript, generate a structured summary, review it with guardrail feedback, then save it to the output folder.**

An end-to-end pipeline that transforms raw insurance call transcripts into structured, guardrail-validated summaries using a Groq-hosted LLM (`llama-3.1-8b-instant`), exposed through a FastAPI backend and a Streamlit web UI.

> **Note — UI scope:** The Streamlit interface is a functional reviewer tool, not a production-grade UI. The primary engineering focus of this project is the pipeline correctness, guardrail accuracy, observability, and security of the backend service.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Architecture](#project-architecture)
3. [Algorithm and Pipeline](#algorithm-and-pipeline)
4. [Project Structure](#project-structure)
5. [Components](#components)
6. [Installation (using uv)](#installation-using-uv)
7. [Configuration](#configuration)
8. [LangSmith Observability](#langsmith-observability)
9. [Running the Application](#running-the-application)
10. [API Reference](#api-reference)
11. [Output Format](#output-format)
12. [Input Guardrails](#input-guardrails)
13. [Output Guardrails](#output-guardrails)
14. [Evaluation System](#evaluation-system)
15. [Security Practices](#security-practices)
16. [Performance Metrics](#performance-metrics)
17. [Running Tests](#running-tests)
18. [Batch Evaluation](#batch-evaluation)
19. [Logging](#logging)
20. [Future Expansion](#future-expansion)

---

## Overview

Insurance claims handlers review dozens of call recordings per shift. This agent automates the post-call summary step: a handler uploads a `.txt` transcript, the system validates it through a three-tier input guardrail suite, calls the LLM, validates the output through a three-tier output guardrail suite, scores the result across eight quality dimensions, and saves a standardised summary ready for the claims system.

**Key capabilities:**

- Structured summary generation in a fixed schema (Caller, Subject, Executive Summary, Next Steps, five optional conditional sections)
- Three-tier **input** guardrail: token budget (Tier 1), prompt injection scan — OWASP LLM01 (Tier 2), PII audit — GDPR Article 30 (Tier 3)
- Three-tier **output** guardrail: structural validation (Tier 1 — blocking), format compliance (Tier 2 — advisory), content integrity vs. transcript (Tier 3 — advisory)
- Auto-retry loop (up to 2 retries) on Tier-1 output guardrail failures with targeted corrective prompt addendum
- Agentic eval-feedback loop: if quality grade is below A, the evaluation findings are fed back to the LLM as a corrective prompt for one further attempt
- Eight-dimension deterministic quality evaluation engine with weighted scoring and letter grades (A/B/C/F)
- LangSmith observability with per-stage trace spans (input validation → LLM call → output guardrails → evaluation)
- Streamlit web UI + REST API + CLI entry point
- Sliding-window server-side rate limiter (30 req/min, matching Groq's RPM cap)

---

## Project Architecture

```
  +------------------------------------------------------------------------+
  |                         User Interfaces                                |
  |                                                                        |
  |  +------------------+   +-------------------+   +------------------+  |
  |  |  Streamlit UI    |   |  FastAPI (REST)   |   |  CLI (main.py)   |  |
  |  |  ui/app.py       |   |  api/app.py       |   |  Direct import   |  |
  |  +--------+---------+   +--------+----------+   +--------+---------+  |
  +-----------|----------------------|------------------------|------------+
              |  HTTP (requests)     |  HTTP (FastAPI)        |  Python
              +-----------+----------+------------------------+
                          |
          +---------------v-----------------------------------------+
          |                     Service Layer                        |
          |              call_summarizer/service.py                  |
          |  generate_summary_from_content()  process_directory()   |
          +---+------------------+------------------+---------------+
              |                  |                  |
    +---------v--------+ +-------v--------+ +-------v---------+
    |  Input           | |  LLM Engine    | |  Output         |
    |  Guardrails      | |  summarizer.py | |  Guardrails     |
    |  input_guardrails| |  Groq API      | |  guardrails/    |
    |                  | |  + retry loop  | |                 |
    |  T1 Token Budget | |                | |  T1 Structure   |
    |  T2 Injection    | |  @traceable    | |  T2 Format      |
    |  T3 PII Audit    | |                | |  T3 Content     |
    |  @traceable      | |                | |  @traceable     |
    +------------------+ +-------+--------+ +-------+---------+
                                 |                  |
                         +-------v------------------v--------+
                         |      Evaluation Engine            |
                         |      evaluator.py                 |
                         |  8-metric quality scoring         |
                         |  + agentic feedback loop          |
                         |  @traceable                       |
                         +-----------------------------------+
                                         |
                         +---------------v---------------+
                         |      LangSmith Traces         |
                         |  validate_input               |
                         |  generate_summary (LLM span)  |
                         |  run_output_guardrails        |
                         |  evaluate_summary             |
                         +-------------------------------+
```

### Data Flow

```
  Upload .txt transcript
        |
        v
  Input Guardrails ---------- BLOCKED? -------> HTTP 400 / error returned
        | (allowed)
        v
  LangGraph Pipeline
        |  +----------+     +------------+     +--------+
        +->|  load    |---->|  summarize |---->|  save  |
           +----------+     +------------+     +--------+
                                   |
                              Groq LLM call
                              (max 600 tokens)
                                   |
                            Output Guardrails
                            T1 fail? --> retry (max 2x, targeted prompt)
                                   |
                            Evaluation Engine
                            Grade < A? --> eval-feedback retry (max 1x)
                                   |
                            Return summary + grade to API
                                   |
                  +----------------v------------------+
                  |   Reviewer in Streamlit UI        |
                  |   (see guardrail findings + grade)|
                  +----------------+------------------+
                                   |  Submit & Save
                                   v
                         POST /api/v1/summaries
                         (re-runs T1 guardrails)
                                   |
                                   v
                         Output_data/<name>-summary.txt
```

---

## Algorithm and Pipeline

### LangGraph Pipeline (3 nodes)

```
  State: { transcript_path, transcript_content, summary, output_path, error }

  +----------+          +---------------+          +----------+
  |  [load]  |--------->|  [summarize]  |--------->|  [save]  |
  +----------+          +---------------+          +----------+
  Reads .txt            Calls Groq LLM             Writes to
  file into state       + output guardrails         Output_data/
                        + auto-retry (max 2x)
```

### Two-Phase Generation Loop

```
  Phase 1 — Output Guardrail Loop (max 3 total attempts)
  -------------------------------------------------------
  generate_summary(transcript)
        |
  run_guardrails(summary)
        |
   T1 errors?  YES --> build_retry_prompt_addendum() --> retry (up to 2x)
               NO  --> Phase 2

  Phase 2 — Eval-Feedback Agentic Loop (max 2 total attempts)
  ------------------------------------------------------------
  evaluate_summary(summary, transcript)
        |
   Grade < A?  YES --> build_eval_feedback_prompt() --> regenerate (1x)
               NO  --> return result
```

### Token Budget Calculation

```
  Groq llama-3.1-8b-instant — 6,000 tokens / minute

  System prompt tokens  ~  len(SYSTEM_PROMPT) / 4  ~  465
  Human prefix tokens   ~  len("Summarise this...") / 4  ~  8
  Fixed overhead        =  473
  Max output tokens     =  600
  ─────────────────────────────────────────────────────────
  Max transcript tokens =  6,000 - 473 - 600  =  4,927
  Max transcript chars  =  4,927 × 4          =  19,708
```

This budget is enforced by the Tier-1 input guardrail before the LLM is called, leaving quota headroom for up to 2 automatic retries in the same 60-second window.

---

## Project Structure

```
Call_summarizer_agent/
├── call_summarizer/              Core package
│   ├── config.py                 App configuration (env vars → Config dataclass)
│   ├── models.py                 Shared data classes (Finding, GuardrailResult, ProcessingResult)
│   ├── service.py                Orchestration layer (used by all callers)
│   ├── summarizer.py             LLM client, system prompt, retry logic
│   ├── evaluator.py              8-metric quality evaluation engine
│   ├── graph.py                  LangGraph pipeline (load → summarize → save)
│   │
│   ├── guardrails/               Output guardrail engine
│   │   ├── constants.py          Schema names, domain terms, regex patterns
│   │   ├── helpers.py            Section + entity extraction helpers
│   │   ├── tier1_structural.py   Blocking structural error checks
│   │   ├── tier2_format.py       Advisory format quality checks
│   │   ├── tier3_content.py      Advisory transcript cross-checks
│   │   ├── runner.py             run_guardrails() + build_retry_prompt_addendum()
│   │   └── __init__.py           Public API + internal re-exports
│   │
│   ├── input_guardrails/         Input guardrail engine
│   │   ├── constants.py          Token budget math, injection patterns, PII regexes
│   │   ├── models.py             InputFinding, InputValidationResult
│   │   ├── tier1_token_budget.py _check_token_budget() — BLOCKING
│   │   ├── tier2_injection.py    _check_injection() — BLOCKING (OWASP LLM01)
│   │   ├── tier3_pii.py          _audit_pii() — NON-BLOCKING (GDPR Article 30)
│   │   ├── runner.py             validate_transcript_input()
│   │   └── __init__.py           Public API
│   │
│   ├── observability/            Logging and tracing
│   │   ├── logging.py            Rotating file + console log handler setup
│   │   ├── tracing.py            LangSmith @traceable span wrappers
│   │   └── __init__.py
│   │
│   └── utils/                    Shared utilities
│       ├── storage.py            save_summary(), derive_output_path()
│       ├── transcript.py         load_transcript(), find_transcripts()
│       ├── validator.py          validate_input_file(), validate_summary()
│       └── __init__.py
│
├── api/                          FastAPI REST backend
│   ├── app.py                    App factory, lifespan, /health endpoint
│   ├── schemas.py                Pydantic request/response models
│   ├── routes/summarize.py       POST /api/v1/summarize, POST /api/v1/summaries
│   └── middleware/rate_limiter.py  Sliding-window rate limiter (30 req/min)
│
├── ui/                           Streamlit reviewer UI
│   └── app.py
│
├── scripts/
│   └── generate_and_evaluate.py  Batch generation + quality report
│
├── tests/                        Unit + integration tests (294 tests)
├── Sample_data/                  Example transcripts and target evaluation set
├── main.py                       CLI entry point
├── run.py                        Launcher (starts FastAPI + Streamlit together)
├── .env.example                  Template — copy to .env and fill in credentials
└── pyproject.toml
```

---

## Components

| Component | File(s) | Responsibility |
|---|---|---|
| Config | `config.py` | Reads env vars; validates `GROQ_API_KEY` |
| LLM Engine | `summarizer.py` | Builds Groq client; manages prompt; exponential-backoff retry |
| LangGraph | `graph.py` | Three-node DAG: load → summarize → save |
| Service | `service.py` | Orchestrates both retry loops; single entry point for all callers |
| Input Guardrails | `input_guardrails/` | Validates transcript before LLM call |
| Output Guardrails | `guardrails/` | Validates summary after LLM call |
| Evaluator | `evaluator.py` | 8-metric quality scoring + agentic feedback prompt builder |
| Observability | `observability/` | Rotating log handler + LangSmith trace spans |
| Utils | `utils/` | File I/O: load transcript, save summary, basic validation |
| REST API | `api/` | FastAPI endpoints + rate limiter middleware |
| UI | `ui/app.py` | Streamlit reviewer (upload → preview → save) |

---

## Installation (using uv)

```bash
# 1. Clone the repository
git clone <repository-url>
cd Call_summarizer_agent

# 2. Install uv (if not already installed)
pip install uv

# 3. Create virtual environment and install dependencies
uv sync

# 4. Configure credentials
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY (and optionally LANGSMITH_API_KEY)
```

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env`:

```ini
# ── Groq API (required) ────────────────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key_here

# ── LangSmith Observability (optional) ────────────────────────────────────
LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# EU endpoint: https://eu.api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=Call summarizer agent
```

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API authentication key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model identifier |
| `INPUT_DIR` | `Input_data` | Directory scanned for `.txt` transcripts (CLI/batch) |
| `OUTPUT_DIR` | `Output_data` | Directory where summaries are saved |
| `LANGSMITH_TRACING` | `false` | Set to `true` to enable LangSmith tracing |
| `LANGSMITH_API_KEY` | — | LangSmith authentication key |
| `LANGSMITH_PROJECT` | `<default>` | LangSmith project name |
| `LANGSMITH_ENDPOINT` | US endpoint | Override with EU endpoint if required |

---

## LangSmith Observability

When `LANGSMITH_TRACING=true`, the pipeline emits **four nested trace spans** per request so you can isolate exactly which stage is slow or producing low-quality output:

```
[call_summarizer_pipeline]             ← top-level run in LangSmith UI
  ├── [validate_input]                 ← T1/T2/T3 input checks
  ├── [generate_summary]  (LLM span)  ← Groq API call + retries
  │     └── groq.chat.completions     ← auto-traced by LangChain
  ├── [run_output_guardrails]          ← T1/T2/T3 output checks
  └── [evaluate_summary]              ← 8-metric quality scoring
```

Without explicit spans, LangSmith would see the entire pipeline as a single undifferentiated block. The `@traceable` decorator in `observability/tracing.py` is a transparent no-op when tracing is disabled, so there is zero overhead in non-tracing deployments.

**What each span records:**
- **validate_input** — allowed/blocked decision, PII categories found
- **generate_summary** — token counts, model used, attempt number
- **run_output_guardrails** — passed/failed, error codes, warning count
- **evaluate_summary** — overall score, grade, per-metric breakdown

---

## Running the Application

### Start everything (recommended)

```bash
uv run run.py
```

This launches the FastAPI backend on **port 8000** and the Streamlit UI on **port 8501** in a single command.

Expected startup output:

```
[run.py] LangSmith tracing: ENABLED (project: 'Call summarizer agent')
[run.py] Starting FastAPI backend on :8000 ...
[run.py] Starting Streamlit UI on :8501 ...
[run.py] Both services are running.
         API docs : http://localhost:8000/docs
         UI       : http://localhost:8501
         Press Ctrl+C to stop.
```

### Start services individually

```bash
# FastAPI only
uv run uvicorn api.app:app --reload --port 8000

# Streamlit only
uv run streamlit run ui/app.py --server.port 8501
```

### CLI (batch processing)

```bash
uv run python main.py
```

Processes all `.txt` files found in `INPUT_DIR` (default `Input_data/`) and saves summaries to `OUTPUT_DIR` (default `Output_data/`).

---

## API Reference

### `POST /api/v1/summarize`

Upload a `.txt` transcript and receive a generated summary with full guardrail and evaluation results. The summary is **not saved** — call `/api/v1/summaries` to persist after review.

**Request:** `multipart/form-data` with a single `file` field (`.txt` only).

**Response `200`:**

```json
{
  "filename": "7-transcript",
  "summary": "Caller: ...\n\nSubject: ...",
  "char_count": 892,
  "within_char_limit": true,
  "passed_guardrails": true,
  "errors": [],
  "warnings": ["'Vehicle Damage' section is included but no related terms found"],
  "eval_grade": "A",
  "eval_score": 0.95,
  "eval_findings": []
}
```

**Error codes:**

| Code | Reason |
|---|---|
| 400 | Non-`.txt` file, not UTF-8, empty, transcript too long, injection detected |
| 429 | Rate limit exceeded (30 req/min on `/api/v1/summarize`) |
| 500 | LLM call failed after all retries |

### `POST /api/v1/summaries`

Save a (possibly user-edited) summary to `Output_data/`. Re-runs Tier-1 guardrails before writing.

**Request body (JSON):**

```json
{ "filename": "7-transcript", "summary": "Caller: ..." }
```

**Response `201`:**

```json
{
  "output_filename": "7-transcript-summary.txt",
  "output_path": "Output_data/7-transcript-summary.txt",
  "char_count": 892,
  "warnings": [],
  "message": "Summary saved successfully to 7-transcript-summary.txt"
}
```

### `GET /health`

Liveness check. Returns `{"status": "ok"}` with HTTP 200.

---

## Output Format

```
Caller: [Name], [relationship], [inbound/outbound]

Subject:
[One-line description of the call topic]

Executive Summary:
[One paragraph narrative of what happened]
- [Key fact extracted from the call]
- [Key fact extracted from the call]

Next Steps:
[Company Name]: [Action or "None"]
Other: [Third-party action or "None"]

--- Conditional sections (include only when topic was discussed) ---

Liability Summary:    [Include if liability discussed]
Negotiation Summary:  [Include if negotiation occurred]
Vehicle Damage:
  Vehicle Status: [Status]
  Towage: [Details or "None"]
  Car hire: [Details or "None"]
Injury:
  Treatment: [Details]
Property:
  [Details]
```

Maximum **1,500 characters** including all whitespace.

---

## Input Guardrails

Applied to every uploaded transcript **before** the LLM call.

| Tier | Name | Blocking | Code | Description |
|---|---|---|---|---|
| 1 | Token Budget | YES | `TRANSCRIPT_TOO_LONG` | Rejects transcripts > 19,708 chars (~4,927 tokens) — prevents quota exhaustion |
| 2 | Injection Scan | YES | `PROMPT_INJECTION_DETECTED` | 14 OWASP LLM01 patterns; tuned to avoid false positives on insurance language |
| 3 | PII Audit | NO | `PII_DETECTED` | Detects email, IBAN, phone, postcode, DOB; writes GDPR Article 30 audit log |

---

## Output Guardrails

Applied to the LLM output after every generation attempt.

### Tier 1 — Structural Errors (blocking)

| Code | Description |
|---|---|
| `EMPTY_SUMMARY` | Summary is blank |
| `CHAR_LIMIT_EXCEEDED` | Summary > 1,500 characters |
| `MISSING_SUBJECT` | Required Subject section absent |
| `MISSING_EXECUTIVE_SUMMARY` | Required Executive Summary absent |
| `MISSING_NEXT_STEPS` | Required Next Steps section absent |
| `SUBJECT_MULTILINE` | Subject spans more than one line |
| `EXECUTIVE_SUMMARY_NO_BULLETS` | Executive Summary has no `- bullet` lines |
| `NEXT_STEPS_INCOMPLETE` | Missing company action or Other line |
| `PHANTOM_CONDITIONAL_SECTION` | Conditional section set to `None` instead of omitted |
| `UNKNOWN_SECTION_HEADER` | Section header not in the defined schema |
| `CONDITIONAL_SECTION_EMPTY_BODY` | Conditional section present but empty |

### Tier 2 — Format Warnings (advisory)

| Code | Description |
|---|---|
| `MISSING_CALLER_LINE` | Caller line absent |
| `CALLER_DIRECTION_MISSING` | `inbound` / `outbound` not in Caller line |
| `CALLER_RELATIONSHIP_UNRECOGNIZED` | Relationship not a known value |
| `NEXT_STEPS_BOTH_NONE` | Both Next Steps actions are "None" |
| `CHAR_COUNT_HIGH` | 1,200–1,500 chars — approaching limit |
| `DUPLICATE_BULLET_CONTENT` | Same numeric value in multiple bullets |
| `VEHICLE_DAMAGE_TOWAGE_MISSING` | Vehicle Damage present but Towage sub-field absent |

### Tier 3 — Content Integrity Warnings (advisory, requires transcript)

| Code | Description |
|---|---|
| `AMOUNT_NOT_IN_TRANSCRIPT` | Currency amount in summary not found in transcript |
| `REFERENCE_NOT_IN_TRANSCRIPT` | Claim/policy reference not in transcript |
| `IBAN_NOT_IN_TRANSCRIPT` | IBAN in summary cannot be matched to transcript |
| `EMAIL_NOT_IN_TRANSCRIPT` | Email address in summary not found in transcript |
| `UNVERIFIED_CONFIRMATION` | Confirmation phrase unsupported by transcript evidence |
| `CONDITIONAL_SECTION_UNJUSTIFIED` | Conditional section included but no domain terms in transcript |

---

## Evaluation System

Eight deterministic metrics scored 0.0–1.0, then weighted into an overall score.

| Metric | Weight | What it checks |
|---|---|---|
| Factual Groundedness | 30% | Every amount, IBAN, email, reference in the summary exists in the transcript |
| Completeness | 20% | Key facts in the transcript are captured in the summary (recall) |
| Format Compliance | 20% | Required sections present, correct structure, within character limit |
| Hallucination | 10% | Confirmation phrases ("confirmed bank details", "waived consideration period") supported by transcript evidence |
| Professionalism | 5% | No informal markers, jargon (TBD/FYI), placeholders, or rogue all-caps |
| Handoff Readiness | 5% | Actionable Next Steps, descriptive Subject, substantive Executive Summary, at least one verifiable identifier |
| Section Precision | 5% | Conditional sections only when topic was discussed |
| Redundancy | 5% | Same fact not repeated across multiple bullets |

### Grade Scale

| Grade | Score | Interpretation |
|---|---|---|
| **A** | ≥ 90% | Production-ready — save without review concerns |
| **B** | ≥ 75% | Minor issues — usable with reviewer awareness |
| **C** | ≥ 60% | Notable gaps — needs correction before filing |
| **F** | < 60% | Significant errors — retry or escalate |

---

## Security Practices

This project was designed with a security-first mindset at every layer of the pipeline.

### OWASP LLM Top 10 Compliance

| Risk | Mitigation |
|---|---|
| **LLM01 — Prompt Injection** | Tier-2 input guardrail scans all transcripts with 14 regex patterns before any content reaches the LLM. Patterns are tuned to avoid false positives on legitimate insurance language (e.g. "act as a witness" does NOT match). |
| **LLM02 — Insecure Output Handling** | Tier-1 output guardrail rejects structurally invalid summaries before they are presented to the UI or saved. The Streamlit UI performs no HTML rendering of LLM output. |
| **LLM06 — Sensitive Information Disclosure** | PII is audited at Tier-3 input guardrail (GDPR Article 30 compliant logging). Transcripts are never stored server-side; they are processed in-memory only. |
| **LLM09 — Overreliance** | Tier-3 content integrity checks cross-reference every verifiable fact (amounts, IBANs, references, emails) against the source transcript before the summary is presented. |

### API Security

- **Rate limiting:** Sliding-window rate limiter (30 req/min, matching Groq's RPM cap) is enforced at the middleware layer before any request handler executes. Returns `429` with `Retry-After` and `X-RateLimit-*` headers.
- **Input validation:** File extension (`.txt` only), encoding (UTF-8), and content length are validated before the transcript reaches any processing logic.
- **CORS:** Restricted to `http://localhost:8501` (Streamlit origin) in development. Override `_STREAMLIT_ORIGIN` in `api/app.py` for production deployment.
- **Secrets management:** All credentials live in `.env` (git-ignored). `.env.example` ships with placeholder values only. `GROQ_API_KEY` is validated at startup — the server refuses to start if it is missing or still the placeholder value.

### Data Handling

- **No persistence of transcripts:** Uploaded transcript content is held in memory only for the duration of the request. Nothing is written to disk except the generated summary (to `Output_data/`).
- **PII audit log:** The Tier-3 PII audit writes structured log entries (category names only — no actual PII values) to the rotating log file, satisfying NIST AI RMF MEASURE 2.5.
- **Hallucination containment:** The Tier-3 content integrity checks and the Hallucination evaluation metric both detect unverified confirmation phrases. The agentic feedback loop will attempt to correct hallucinations before the summary is surfaced to the reviewer.

---

## Performance Metrics

Measured on a batch of 10 target transcripts (`Sample_data/target_data/`) against Groq `llama-3.1-8b-instant`. All timings are wall-clock including network round trips.

### Quality Results (latest batch run)

| Metric | Score |
|---|---|
| **Overall average** | **94.3%** |
| Grade A | 10 / 10 (100%) |
| Grade B | 0 / 10 |
| Factual Groundedness | 100.0% |
| Completeness | 81.0% |
| Format Compliance | 96.9% |
| Hallucination | 100.0% |
| Professionalism | 97.5% |
| Handoff Readiness | 98.0% |
| Section Precision | 82.0% |
| Redundancy | 97.5% |

> Completeness (81%) and Section Precision (82%) are the two weakest metrics. Both reflect the LLM occasionally omitting minor call details or including conditional sections without strong transcript support. The agentic eval-feedback loop corrects these in the majority of cases (moved one Grade B to Grade A in the last run).

### API Latency (offline mock tests — `tests/test_api_performance.py`)

Latency is measured against a mocked LLM (`TestClient`, no network I/O) to isolate the application overhead from Groq API variability.

| Percentile | Threshold | Typical |
|---|---|---|
| Mean | < 100 ms | ~15 ms |
| p90 | < 150 ms | ~25 ms |
| p95 | < 200 ms | ~35 ms |
| p99 | < 500 ms | ~60 ms |

### LLM Call Latency (live, single transcript, no retries needed)

| Stage | Typical |
|---|---|
| Input guardrails | < 5 ms |
| Groq LLM call | 1 – 4 s |
| Output guardrails | < 10 ms |
| Evaluation engine | < 15 ms |
| **Total end-to-end** | **1 – 5 s** |

### Rate Limits

| Limit | Value | Enforced by |
|---|---|---|
| Groq TPM | 6,000 tokens/min | Input Tier-1 guardrail (token budget) |
| Groq RPM | 30 requests/min | Server-side sliding-window rate limiter |
| Summary max length | 1,500 chars | Output Tier-1 guardrail + system prompt |
| Transcript max length | 19,708 chars | Input Tier-1 guardrail |

---

## Running Tests

```bash
# All tests
uv run python -m pytest tests/ -v

# By category
uv run python -m pytest tests/test_guardrails.py -v         # Output guardrail checks
uv run python -m pytest tests/test_input_guardrails.py -v   # Input guardrail checks
uv run python -m pytest tests/test_api_performance.py -v    # API latency + rate limiting
uv run python -m pytest tests/test_storage.py -v            # File I/O utilities
```

### Test Coverage Summary

| Test file | Tests | Coverage area |
|---|---|---|
| `test_guardrails.py` | ~200 | All 18 output guardrail check codes |
| `test_input_guardrails.py` | ~50 | All 3 input guardrail tiers, all 14 injection patterns |
| `test_api_performance.py` | 22 | Latency percentiles, rate-limit enforcement, LLM retry behaviour |
| `test_storage.py` | ~10 | `save_summary`, `derive_output_path` |
| `test_transcript.py` | ~10 | `load_transcript`, `find_transcripts` |
| `test_validator.py` | ~10 | `validate_input_file`, `validate_summary` |
| **Total** | **294** | |

---

## Batch Evaluation

Run the full pipeline on the 10 target transcripts and print a quality report:

```bash
uv run python scripts/generate_and_evaluate.py
```

```
Generating summaries for 10 transcript(s) in Sample_data/target_data/

  [ 1/10] 1-transcript.txt ... OK (1.3s)
  ...
  [10/10] 9-transcript.txt ... OK (25.6s)

====================================================================
                         EVALUATION REPORT
====================================================================
File                   Gr       Score  Ground   Compl  Format   ...
--------------------------------------------------------------------
1-transcript           A        95.0%    100%    100%    100%   ...
...
====================================================================

  Total   : 10 summaries
  Average : 94.3%
  Grade A : 10  **********
```

Options:

```bash
# Skip generation — evaluate already-saved summaries in Output_data/
uv run python scripts/generate_and_evaluate.py --eval-only

# Verbose: show detailed per-metric findings for every summary
uv run python scripts/generate_and_evaluate.py --verbose

# Custom directories
uv run python scripts/generate_and_evaluate.py \
  --transcripts-dir Sample_data/target_data \
  --output-dir Output_data
```

---

## Logging

All components use Python's standard `logging` module with a module-level logger (`logging.getLogger(__name__)`). The logging system is initialised in `call_summarizer/observability/logging.py`.

```
logs/call_summarizer.log    — primary log file
logs/call_summarizer.log.1  — previous rotation
logs/call_summarizer.log.2  — two rotations ago
logs/call_summarizer.log.3  — three rotations ago (oldest kept)
```

- **Max file size:** 5 MB per file
- **Backup count:** 3
- **Format:** `2024-01-15 14:23:01 | INFO     | module.function | message`
- **Console:** Same format, same level, written to stderr

Key log entries to watch:

| Pattern | Meaning |
|---|---|
| `LangSmith tracing ENABLED` | Traces are being sent to LangSmith |
| `[INPUT-T1]` | Token budget check result |
| `[INPUT-T2]` | Injection scan result |
| `[PII AUDIT]` | PII categories detected |
| `[GUARDRAIL][ERROR]` | Tier-1 blocking error found |
| `[GUARDRAIL][WARN]` | Tier-2/3 advisory warning |
| `Eval [phase-2 attempt` | Evaluation-feedback loop iteration |
| `Generation complete` | Final grade and attempt count for a transcript |

---

## Future Expansion

The following capabilities are planned or recommended for a production deployment.

### Caching and Deduplication

A content-addressable cache using a **SHA-256 hash of the transcript** would prevent redundant LLM calls for re-uploaded transcripts:

```
transcript → SHA-256 hash → lookup in Redis/DynamoDB
    HIT  → return cached summary (0 ms, 0 tokens)
    MISS → run pipeline → store result keyed by hash
```

This is especially valuable for the batch processing path where the same transcript may be re-processed after a schema change. The cache key should include a version string tied to the system prompt so stale summaries are invalidated when the prompt changes.

### Database Integration

Replacing file-based storage with a relational database enables:

- **Audit trail:** Every summary version, generation attempt, guardrail findings, and evaluation grade stored with timestamps and user identity.
- **Search and retrieval:** Query summaries by claim reference, caller name, date range, or quality grade.
- **API association:** A `GET /api/v1/summaries/{claim_id}` endpoint backed by the database would let the claims system pull summaries directly rather than reading files from disk.

Suggested schema:

```sql
summaries (
    id UUID PRIMARY KEY,
    claim_reference TEXT,
    transcript_hash TEXT,        -- SHA-256, for cache dedup
    summary_text TEXT,
    eval_grade CHAR(1),
    eval_score REAL,
    guardrail_passed BOOL,
    created_at TIMESTAMPTZ,
    created_by TEXT
)
```

### LLM-as-Judge Evaluation for Production

The current evaluation engine is fully **deterministic** (regex + entity extraction). For a production environment, a secondary LLM judge (e.g. `claude-opus-4` or `gpt-4o`) can evaluate dimensions that are hard to assess deterministically:

- **Tone and appropriateness:** Is the language suitable for external communication with a customer or auditor?
- **Contextual completeness:** Has anything factually important been omitted that was not captured by entity extraction?
- **Handoff quality:** Would an unfamiliar agent understand the full picture from this summary alone?

This is a **hybrid approach** — keep the deterministic checks (fast, interpretable, zero cost) as Tier-1/2/3 guardrails, and add an LLM judge as a post-generation quality gate only for summaries that will be filed without human review.

LangSmith's evaluation framework supports this pattern natively via `langsmith.evaluate()`.

### PII Redaction Strategy

The current Tier-3 input guardrail **audits** PII but does not remove it — insurance transcripts legitimately require PII to produce accurate summaries. For use cases where the LLM must not see raw PII (e.g. a shared or third-party LLM endpoint), a **mask-then-restore** approach:

```
Transcript (raw)
    |
    v
PII Detector → extract PII entities → store in context map
    |                                 { "IBAN_1": "IE29AIBK...",
    |                                   "PHONE_1": "087-123-4567" }
    v
Masked transcript
("... transfer to IBAN_1 ... call back on PHONE_1 ...")
    |
    v
LLM call (never sees real PII values)
    |
    v
Masked summary
    |
    v
PII Restore → replace tokens with original values
    |
    v
Final summary (with real values, same accuracy)
```

Libraries: Microsoft Presidio (`presidio-analyzer` + `presidio-anonymizer`) or AWS Comprehend Medical for clinical/insurance PII.

### Map-Reduce Chunking for Long Transcripts

The current Tier-1 input guardrail blocks transcripts exceeding ~19,708 characters. For organisations that handle longer calls (30–60 minutes), a **map-reduce summarisation strategy**:

```
Long transcript (> 19,708 chars)
    |
    v
Chunk into overlapping windows
(e.g. 4,000 chars each, 200 char overlap)
    |
    v [Map phase]
Summarise each chunk independently
    |
    v [Reduce phase]
Merge chunk summaries with a second LLM call
("Given these partial summaries, produce one final summary")
    |
    v
Run full guardrail + evaluation suite on final summary
```

The overlap prevents facts that span a chunk boundary from being lost. The reduce prompt should specify that all unique facts from each partial summary must be preserved, and the same 1,500-character output schema must be respected.
