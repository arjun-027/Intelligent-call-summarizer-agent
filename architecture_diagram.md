# Project Architecture Diagram

This document provides a detailed layout of the **Insurance Call Summariser Agent** architecture. It maps the interfaces, service boundaries, LangGraph execution nodes, guardrails, and file flows to show how the components interact.

## Architectural Diagram

Here is a visual mapping of the system components and their relationships:

```mermaid
graph TD
    %% Styling Definitions
    classDef uiStyle fill:#e0f2fe,stroke:#0284c7,stroke-width:1.5px;
    classDef apiStyle fill:#ecfdf5,stroke:#059669,stroke-width:1.5px;
    classDef serviceStyle fill:#fffbeb,stroke:#d97706,stroke-width:1.5px;
    classDef graphStyle fill:#faf5ff,stroke:#7c3aed,stroke-width:1.5px;
    classDef ioStyle fill:#fdf2f8,stroke:#db2777,stroke-width:1.5px;

    %% UI Layer
    subgraph UI ["User Interfaces"]
        streamlitApp["Streamlit Frontend<br/>(ui/app.py)"]:::uiStyle
        cliApp["CLI Entrypoint<br/>(main.py)"]:::uiStyle
        runnerApp["Service Runner CLI<br/>(run.py)"]:::uiStyle
    end

    %% API Layer
    subgraph API ["FastAPI Service Layer"]
        fastapiApp["FastAPI App Server<br/>(api/app.py)"]:::apiStyle
        schemas["Pydantic schemas<br/>(api/schemas.py)"]:::apiStyle
        routes["HTTP Router endpoints<br/>(api/routes/summarize.py)"]:::apiStyle
    end

    %% Core Services Layer
    subgraph Services ["Core Logic & Service Layer"]
        config["Configuration dataclass<br/>(call_summarizer/config.py)"]:::serviceStyle
        service["Service Orchestrator<br/>(call_summarizer/service.py)"]:::serviceStyle
        inputGuard["Input Guardrails Engine<br/>(call_summarizer/input_guardrails.py)"]:::serviceStyle
        outputGuard["Output Guardrails Engine<br/>(call_summarizer/guardrails.py)"]:::serviceStyle
        evaluator["Deterministic Quality Evaluator<br/>(call_summarizer/evaluator.py)"]:::serviceStyle
    end

    %% LangGraph Pipeline
    subgraph GraphPipeline ["LangGraph Execution Pipeline (call_summarizer/graph.py)"]
        stateDef["SummaryState definition<br/>(call_summarizer/models.py)"]:::graphStyle
        langgraphCompile["Compiled LangGraph pipeline"]:::graphStyle
        loadNode["Node 1: load_node<br/>(call_summarizer/transcript.py)"]:::graphStyle
        summarizeNode["Node 2: summarize_node<br/>(call_summarizer/summarizer.py)"]:::graphStyle
        saveNode["Node 3: save_node<br/>(call_summarizer/storage.py)"]:::graphStyle
    end

    %% Data and External Interfaces
    subgraph IO ["IO & External Interfaces"]
        envFile[".env config properties"]:::ioStyle
        inputDir["Input_data/ (.txt transcripts)"]:::ioStyle
        outputDir["Output_data/ (*-summary.txt files)"]:::ioStyle
        groqAPI["Groq LLM Client<br/>(llama-3.1-8b-instant)"]:::ioStyle
        logs["Rotating Log Files<br/>(call_summarizer.log)"]:::ioStyle
    end

    %% UI Connections
    runnerApp -->|starts| fastapiApp
    runnerApp -->|starts| streamlitApp
    streamlitApp -->|REST POST /summarize| fastapiApp
    fastapiApp --> routes
    routes --> schemas

    %% Service connections
    cliApp -->|direct run| service
    routes -->|invokes| service
    streamlitApp -->|optional direct run| service
    service --> config
    config -->|reads| envFile

    %% Validation flows
    service -->|checks input| inputGuard
    inputGuard -->|reads token budget / scanning| service
    
    %% LangGraph Execution Pipeline mapping
    service -->|invokes| langgraphCompile
    langgraphCompile -->|uses| stateDef
    langgraphCompile --> loadNode
    loadNode -->|reads transcript| inputDir
    loadNode --> summarizeNode
    summarizeNode -->|calls Groq inference| groqAPI
    
    %% Guardrails and Auto-retry Loop
    summarizeNode -->|validates summary| outputGuard
    service -->|manages retry loop / addendum| outputGuard
    outputGuard -->|structural errors feedback| service
    service -->|retry attempt with prompt addendum| summarizeNode

    summarizeNode --> saveNode
    saveNode -->|writes final summary| outputDir

    %% Quality Evaluation
    service -->|evaluates overall score| evaluator
    service -->|audits / writes logs| logs
```

## Detailed Component Reference

### UI Layer
*   [run.py](file:///f:/bright%20beam_final/Call_summarizer_agent/run.py) starts FastAPI (port 8000) and Streamlit (port 8501) as concurrent processes, capturing termination inputs (`Ctrl+C`).
*   [ui/app.py](file:///f:/bright%20beam_final/Call_summarizer_agent/ui/app.py) handles the 3-step Streamlit web application dashboard (upload, preview validation results, edit/save).
*   [main.py](file:///f:/bright%20beam_final/Call_summarizer_agent/main.py) provides a direct terminal execution path for batch processing.

### API Layer
*   [api/app.py](file:///f:/bright%20beam_final/Call_summarizer_agent/api/app.py) mounts CORS origins and coordinates service lifecycle events.
*   [api/schemas.py](file:///f:/bright%20beam_final/Call_summarizer_agent/api/schemas.py) validates client payloads using [SummaryRequest](file:///f:/bright%20beam_final/Call_summarizer_agent/api/schemas.py) and [SummaryResponse](file:///f:/bright%20beam_final/Call_summarizer_agent/api/schemas.py).
*   `api/routes/summarize.py` manages endpoint routing for the REST layer.

### Core Processing Layer
*   [call_summarizer/service.py](file:///f:/bright%20beam_final/Call_summarizer_agent/call_summarizer/service.py) orchestrates the execution. It handles:
    1. Running input validation via input guardrails.
    2. Invoking the LangGraph pipeline.
    3. Executing the automatic retry loop (up to 2 times) with feedback corrections if Tier-1 structural output guardrails fail.
    4. Saving summaries and running final evaluations.
*   [call_summarizer/input_guardrails.py](file:///f:/bright%20beam_final/Call_summarizer_agent/call_summarizer/input_guardrails.py) checks input safety:
    *   **Tier 1**: Token budget validation (blocks if > 19,708 characters).
    *   **Tier 2**: OWASP prompt injection scanner (blocks on malicious sequences).
    *   **Tier 3**: GDPR PII audit (scans for emails, IBANs, phone numbers, postcodes; logs only).
*   [call_summarizer/guardrails.py](file:///f:/bright%20beam_final/Call_summarizer_agent/call_summarizer/guardrails.py) checks output compliance:
    *   **Tier 1 (Structural)**: Missing headers or format schemas (blocks saving and triggers auto-retry).
    *   **Tier 2 (Format)**: Length and styling anomalies (warning only).
    *   **Tier 3 (Content)**: Verification of amounts, references, and IBANs against source transcript.
*   [call_summarizer/evaluator.py](file:///f:/bright%20beam_final/Call_summarizer_agent/call_summarizer/evaluator.py) performs post-generation evaluation scoring across 8 dimensions (factual grounding, completeness, formatting, hallucinations, professionalism, readiness, section relevance, and redundancy).

### LangGraph Pipeline
*   [call_summarizer/graph.py](file:///f:/bright%20beam_final/Call_summarizer_agent/call_summarizer/graph.py) structures node transitions on a linear graph (`load` ──► `summarize` ──► `save`).
*   [call_summarizer/transcript.py](file:///f:/bright%20beam_final/Call_summarizer_agent/call_summarizer/transcript.py) loads files into pipeline memory.
*   [call_summarizer/summarizer.py](file:///f:/bright%20beam_final/Call_summarizer_agent/call_summarizer/summarizer.py) queries the LLM.
*   [call_summarizer/storage.py](file:///f:/bright%20beam_final/Call_summarizer_agent/call_summarizer/storage.py) persists findings to disk.
