# RCSA Control Description QC Automation

An end-to-end pipeline for automated Quality Control of RCSA (Risk and Control Self Assessment) control descriptions at a US financial institution. Built on **LangGraph StateGraph** with **Claude** as the evaluation engine, **Pydantic** data validation, **LangSmith** tracing for observability, and automated **PowerPoint** report generation.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the E2E Pipeline](#running-the-e2e-pipeline)
- [Pipeline Nodes (Step by Step)](#pipeline-nodes-step-by-step)
- [Prompt System](#prompt-system)
- [Output Files](#output-files)
- [Observability & Tracing](#observability--tracing)
- [QC Evaluation Framework](#qc-evaluation-framework)
- [Project Structure](#project-structure)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        StateGraph Pipeline                      │
│                                                                 │
│  ┌──────────────┐    ┌────────────────────┐    ┌────────────┐  │
│  │  validate_    │    │  evaluate_          │    │  aggregate_ │  │
│  │  input        │ ──>│  controls           │ ──>│  results    │  │
│  │ (pydantic)    │    │ (Claude LLM call)   │    │ (JSON save) │  │
│  └──────────────┘    └────────────────────┘    └─────┬──────┘  │
│                                                       │         │
│                                          ┌────────────┴───┐     │
│                                          │ any failures?  │     │
│                                          └──┬──────────┬──┘     │
│                                         yes │          │ no     │
│                                   ┌─────────▼───┐      │        │
│                                   │  build_      │     END      │
│                                   │  presentations│              │
│                                   │  (PPTX gen)  │              │
│                                   └──────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

The pipeline accepts a list of `{risk_description, control_description}` records, validates them, evaluates each against FRASA and internal QC standards via Claude, aggregates results into summary JSON, and conditionally generates PowerPoint reports for any controls that do not fully meet standards.

---

## Prerequisites

- **Python 3.11+**
- **Anthropic API key** (for Claude access)
- **LangSmith API key** (optional, for tracing/observability)

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repo-url>
   cd rcsa_automation
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   .venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r src/requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:

   ```env
   ANTHROPIC_API_KEY=sk-ant-...

   # Optional: enable LangSmith tracing
   LANGCHAIN_API_KEY=lsv2_...
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=RCSA
   ```

---

## Configuration

Runtime behavior is controlled by `config.yaml` in the project root:

```yaml
# Where to write summary.json, full_results.json, and .pptx files
output_dir: ./output

# What to save
save_summary: true            # summary.json  (pass/fail counts + per-record assessments)
save_presentations: true      # .pptx files   (one per failed control)

# LLM overrides (uncomment to override values from the prompt YAML)
# model: claude-sonnet-4-20250514
# temperature: 0.1
```

| Setting              | Type    | Default                      | Description                                              |
|----------------------|---------|------------------------------|----------------------------------------------------------|
| `output_dir`         | path    | `./output`                   | Directory for all output files (relative to project root)|
| `save_summary`       | bool    | `true`                       | Write `summary.json` with aggregated pass/fail stats     |
| `save_presentations` | bool    | `true`                       | Generate `.pptx` reports for failed controls             |
| `model`              | string  | `claude-sonnet-4-20250514`   | Claude model ID (overrides prompt YAML if set)           |
| `temperature`        | float   | `0.1`                        | LLM temperature (overrides prompt YAML if set)           |

---

## Running the E2E Pipeline

### Quick start (built-in test data)

The pipeline ships with sample records for immediate testing:

```bash
python src/chains/e2e_graph.py
```

This runs two example control descriptions through the full pipeline and prints a summary to stdout. Output files are written to `./output/`.

### Programmatic usage

```python
from src.chains.e2e_graph import build_graph, ControlRecord
from src.configs.config import RunConfig

# Load config from config.yaml (or use defaults)
config = RunConfig.load()

# Build and compile the LangGraph StateGraph
app = build_graph()

# Define input records
records = [
    {
        "risk_description": "Unauthorized access to customer PII...",
        "control_description": "The Data Privacy team reviews access logs quarterly..."
    },
    {
        "risk_description": "Segregation of duties violation...",
        "control_description": "The Access Management team enforces RBAC..."
    }
]

# Run the pipeline
result = app.invoke({
    "config": config,
    "records": records,
})

# Access results
print(result["summary"])           # Aggregated stats
for r in result["qc_results"]:     # Per-record QCResult objects
    print(r.overall_assessment)
    print(r.pptx_path)             # Path to PPTX (if generated)
```

### Input format

Each record must have two non-empty string fields:

```json
{
  "risk_description": "<description of the risk being mitigated>",
  "control_description": "<the control description to evaluate>"
}
```

These are validated at runtime via the `ControlRecord` Pydantic model, which enforces `min_length=1` on both fields.

---

## Pipeline Nodes (Step by Step)

The StateGraph executes the following nodes in sequence:

### 1. `validate_input`

**File:** `src/chains/e2e_graph.py`

- Validates each raw input dict against the `ControlRecord` Pydantic model
- Enforces that both `risk_description` and `control_description` are non-empty strings
- Raises `ValidationError` if any record fails schema validation

### 2. `evaluate_controls`

**File:** `src/chains/e2e_graph.py`

- Loads the `e2e` prompt template from the prompt registry (`prompts/rcsa/e2e.yaml`)
- Initializes a `ChatAnthropic` LLM with the configured model and temperature
- Iterates over each `ControlRecord`, invoking the LLM chain with:
  - `risk_description` and `control_description` from the record
  - `qc_standards`, `good_control_example`, and `bad_control_example` from prompt metadata
- Parses the "Overall Assessment" rating from the LLM's markdown output (`MEETS` / `PARTIALLY MEETS` / `DOES NOT MEET`)
- Returns a list of `QCResult` objects

### 3. `aggregate_results`

**File:** `src/chains/e2e_graph.py`

- Computes pass/fail counts and pass rate from the QC results
- Writes `summary.json` with aggregated statistics (if `save_summary` is enabled)
- Writes `full_results.json` with complete per-record QC output and metadata
- Both files are saved to the configured `output_dir`

### 4. `build_presentations` (conditional)

**File:** `src/chains/e2e_graph.py` (orchestration) + `src/chains/pptx_builder.py` (generation)

- **Only runs** if `save_presentations=true` AND at least one control has a non-`MEETS` assessment
- For each failed control, generates a 6-slide PowerPoint deck:
  1. **Title Slide** -- navy background with control summary and status badge
  2. **Evaluation Context** -- QC framework description + original control text
  3. **QC Table** -- detailed per-criteria ratings (Who/What/When/How/Why/Evidence)
  4. **Overall Assessment** -- KPI cards with key findings
  5. **Before/After Comparison** -- original vs. revised control description
  6. **Next Steps** -- auto-generated action items based on findings
- Files are named: `qc_report_{index}_{slug}_{timestamp}.pptx`

---

## Prompt System

Prompts are managed via a YAML-based registry system.

### Registry

**File:** `prompts/registry.yaml`

Maps prompt IDs to their YAML file paths. The `PromptRegistry` class (`src/prompts/registry.py`) loads this file and provides a `get(prompt_id)` method that returns a `(ChatPromptTemplate, metadata)` tuple.

### Active prompt: `e2e`

**File:** `prompts/rcsa/e2e.yaml`

The primary prompt used by the pipeline. It contains:
- A system prompt with detailed QC evaluation instructions
- Input variables: `risk_description`, `control_description`, `qc_standards`, `good_control_example`, `bad_control_example`
- Metadata: `model`, `temperature`, `version`, `tags`
- Embedded QC standards, plus good and bad control examples for few-shot guidance

### Output structure

The LLM returns exactly four markdown sections (no preamble, no commentary):

1. **Control Description Quality Control Report** -- table with columns: QC Criteria, PASS/FAIL, Rationale, Revision
2. **Overall Assessment Table** -- 3-row summary
3. **Fully Revised Control Description** -- rewritten control (max 5 sentences, with `[PLACEHOLDER]` for missing info)

### Legacy prompt

**File:** `copilot_prompt/control_validation_prompt_revised_v3.txt`

A standalone prompt for manual use in Claude or Copilot UI. Not used by the Python pipeline.

---

## Output Files

All outputs are written to the directory specified by `output_dir` in `config.yaml` (default: `./output/`).

| File                  | Description                                                         |
|-----------------------|---------------------------------------------------------------------|
| `summary.json`        | Aggregated stats: total, passed, failed, pass_rate, per-record details |
| `full_results.json`   | Complete results: per-record QC markdown output + metadata          |
| `qc_report_*.pptx`   | PowerPoint decks for failed controls (one per failing record)       |

### Example `summary.json`

```json
{
  "total": 2,
  "passed": 0,
  "failed": 2,
  "pass_rate": "0%",
  "details": [
    {
      "control": "a leash is used to restrain the dog when outside to prevent it from running aw",
      "assessment": "DOES NOT MEET"
    },
    {
      "control": "On an ongoing, continuous basis, the Access Management team in coordination with",
      "assessment": "DOES NOT MEET"
    }
  ]
}
```

---

## Observability & Tracing

The pipeline integrates with **LangSmith** for full observability of LLM calls.

### Enabling tracing

Set these environment variables in `.env`:

```env
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=RCSA
```

When `LANGCHAIN_API_KEY` is set, tracing activates automatically. The pipeline sets `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_PROJECT=RCSA` as defaults if not already configured.

### What is traced

- Full LLM request/response for each control evaluation
- Prompt template rendering
- StateGraph node execution
- Token usage and latency per invocation

Traces are viewable at [smith.langchain.com](https://smith.langchain.com) under the **RCSA** project.

---

## QC Evaluation Framework

Each control description is evaluated against two complementary standards:

### FRASA Control Requirements (5 elements)

| Element            | Description                                |
|--------------------|--------------------------------------------|
| **Frequency**      | How often the control is performed         |
| **Responsible Party** | Who performs the control                |
| **Activity**       | The specific risk-mitigating action(s)     |
| **Source**          | Information or data sources used           |
| **Action Taken**   | Follow-up actions based on control results |

### Internal QC Standards (6 elements)

| Element       | Description                                                              |
|---------------|--------------------------------------------------------------------------|
| **Who**       | Responsible party or system                                              |
| **What**      | Brief control action (Review, Approve, Authenticate, etc.)              |
| **When**      | Execution timing or frequency                                            |
| **How**       | How control mitigates risk + exception handling                          |
| **Why**       | The risk being mitigated                                                 |
| **Evidence**  | Documentation/system evidence and storage location                       |

### Rating scale

| Rating               | Meaning                                           |
|----------------------|---------------------------------------------------|
| `MEETS`              | Element is explicitly and adequately addressed     |
| `PARTIALLY MEETS`    | Element is present but incomplete or vague         |
| `DOES NOT MEET`      | Element is missing or cannot be evaluated          |

All evaluations are **evidence-anchored**: the LLM must cite exact text from the input. If an element cannot be evaluated from the provided text, it must be rated `DOES NOT MEET`.

---

## Project Structure

```
rcsa_automation/
├── .env                              # API keys (not committed)
├── .gitignore
├── config.yaml                       # Runtime configuration
├── CLAUDE.md                         # Claude Code project instructions
├── README.md                         # This file
│
├── prompts/                          # Prompt definitions (YAML)
│   ├── registry.yaml                 # Prompt ID -> file path mapping
│   ├── rcsa/
│   │   ├── e2e.yaml                  # Active pipeline prompt
│   │   └── control_description_qc.yaml  # Legacy single-input prompt
│   └── presentation/
│       └── build_ppt.yaml            # PPTX generation spec (legacy)
│
├── copilot_prompt/                   # Standalone prompt for manual use
│   ├── control_validation_prompt_revised_v3.txt
│   └── trash/                        # Historical prompt versions
│
├── output/                           # Generated outputs
│   ├── summary.json
│   ├── full_results.json
│   └── *.pptx
│
└── src/                              # Python source code
    ├── __init__.py
    ├── requirements.txt
    │
    ├── configs/
    │   └── config.py                 # RunConfig pydantic model
    │
    ├── prompts/
    │   ├── loader.py                 # PromptLoader (YAML -> ChatPromptTemplate)
    │   └── registry.py               # PromptRegistry (ID-based lookup)
    │
    ├── chains/
    │   ├── e2e_graph.py              # LangGraph StateGraph pipeline
    │   ├── pptx_builder.py           # PowerPoint generation & markdown parsing
    │   └── rcsa_control_qc_chain.py  # Legacy chain (not used by e2e)
    │
    ├── notebooks/                    # Development sandbox
    ├── tests/                        # (not yet implemented)
    └── utils/                        # (not yet implemented)
```
