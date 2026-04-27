# Technical Requirements

This document describes the Python packages required to execute the RCSA Control Description QC automation pipeline end-to-end, and explains how each package is used within the project.

---

## Installation

```bash
pip install -r src/requirements.txt
```

---

## Package Reference

### `langchain-core` ≥ 0.3

**Role:** Core abstractions for building LLM-powered chains and prompt templates.

Used in `src/prompts/loader.py` to construct `ChatPromptTemplate` objects from YAML prompt definitions. Each prompt YAML's `system` and `user_template` fields are assembled into a two-message chat template (system + human) that is composed with the LLM via the `|` pipe operator (`prompt | llm`). This is the foundational abstraction that makes prompt definitions portable and reusable across different LLM backends.

---

### `langchain-anthropic` ≥ 0.3

**Role:** LangChain integration for Anthropic's Claude models.

Used in `src/chains/e2e_graph.py` (pipeline entry point) and `src/chains/rcsa_control_qc_chain.py` (legacy chain) to instantiate `ChatAnthropic` — the LLM client that sends QC evaluation requests to Claude. The model ID and temperature are loaded from prompt YAML metadata (e.g., `prompts/rcsa/e2e.yaml`) and can be overridden at runtime via `config.yaml`. The `evaluate_controls` node invokes the chain once per input `ControlRecord`, receiving a structured markdown QC report in return.

---

### `langgraph` ≥ 0.4

**Role:** Stateful multi-step pipeline orchestration using a directed graph.

Used in `src/chains/e2e_graph.py` to define and compile the end-to-end `StateGraph`. The graph wires together five nodes in a typed state machine:

| Node | Description |
|---|---|
| `validate_input` | Validates raw input dicts against the `ControlRecord` Pydantic schema |
| `evaluate_controls` | Calls Claude once per record via the LangChain chain |
| `check_grounding` | Verifies LLM output references key terms from the input (token-overlap grounding check) |
| `aggregate_results` | Computes pass/fail statistics and writes `summary.json` / `full_results.json` |
| `build_presentations` | Generates `.pptx` QC decks for any control that did not fully meet standards |

A conditional edge after `aggregate_results` routes to `build_presentations` only when failures exist and `save_presentations=true`, otherwise the graph terminates. LangGraph manages state propagation between nodes via the `GraphState` TypedDict.

---

### `pydantic` ≥ 2.0

**Role:** Runtime data validation and typed data modeling.

Used throughout the project to enforce schema correctness at system boundaries:

| Model | Location | Purpose |
|---|---|---|
| `ControlRecord` | `e2e_graph.py` | Validates that `risk_description` and `control_description` are non-empty strings |
| `QCResult` | `e2e_graph.py` | Holds per-record evaluation output: control, LLM text, overall rating, PPTX path, grounding result |
| `GroundingResult` | `e2e_graph.py` | Captures token-coverage metrics and grounding pass/fail for each evaluated record |
| `QCRow` / `OverallRow` / `QCReport` | `pptx_builder.py` | Structured representations of parsed LLM markdown output sections used to build slides |
| `RunConfig` | `configs/config.py` | Typed runtime configuration loaded from `config.yaml`; validates `output_dir`, `save_summary`, `save_presentations`, `model`, and `temperature` |

---

### `python-dotenv` ≥ 1.0

**Role:** Loads environment variables from a `.env` file into the process environment.

Called via `load_dotenv()` at module import time in `src/chains/e2e_graph.py` and `src/chains/rcsa_control_qc_chain.py`. This makes the following secrets available to the pipeline without hard-coding them:

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Authenticates requests to the Claude API |
| `LANGCHAIN_API_KEY` | Optional | Enables LangSmith tracing when set |
| `LANGCHAIN_TRACING_V2` | Optional | Activates trace export to LangSmith |
| `LANGCHAIN_PROJECT` | Optional | Groups traces under the `RCSA` project in LangSmith |

---

### `pyyaml` ≥ 6.0

**Role:** Parses YAML files into Python dictionaries.

Used in three places across the project:

- **`src/prompts/loader.py`** — Reads prompt YAML files (e.g., `prompts/rcsa/e2e.yaml`) to extract the `system` prompt, `user_template`, and metadata fields (`model`, `temperature`, `version`, QC standards, few-shot examples).
- **`src/prompts/registry.py`** — Reads `prompts/registry.yaml`, which maps prompt IDs (e.g., `"e2e"`) to their YAML file paths, enabling decoupled prompt lookup.
- **`src/configs/config.py`** — Reads `config.yaml` at the project root to populate the `RunConfig` model with runtime settings.

---

### `python-pptx` ≥ 0.6.21

**Role:** Programmatic creation and formatting of PowerPoint (`.pptx`) files.

Used exclusively in `src/chains/pptx_builder.py` to generate six-slide QC report decks for any control description that does not fully meet standards. The builder parses the LLM's structured markdown output into a `QCReport` model and then constructs each slide using `python-pptx` primitives (`Presentation`, `RGBColor`, `Inches`, `Pt`, `PP_ALIGN`, table shapes, textboxes, and colored rectangles):

| Slide | Content |
|---|---|
| Title | Navy background with control summary and section badges |
| Evaluation Context | QC framework overview and original control description |
| QC Table | Per-criteria MEETS / PARTIALLY MEETS / DOES NOT MEET ratings with rationale and revision suggestions |
| Overall Assessment | Color-coded KPI cards for the overall rating |
| Before / After Comparison | Side-by-side original vs. fully revised control description |
| Next Steps | Auto-generated action items with owner and priority |

Output files follow the naming convention `qc_report_{index}_{slug}_{timestamp}.pptx` and are written to the directory specified by `output_dir` in `config.yaml`.

---

## Standard Library Dependencies

The following Python standard library modules are used by the pipeline and require no separate installation:

| Module | Usage |
|---|---|
| `json` | Serializes `summary.json` and `full_results.json` output files |
| `re` | Parses the LLM's markdown output to extract the Overall Assessment rating and to tokenize text for grounding checks |
| `pathlib` | Cross-platform file and directory path manipulation throughout the codebase |
| `datetime` | Generates timestamps used in `.pptx` output filenames |
| `typing` / `typing_extensions` | Type annotations (`TypedDict`, `List`, `Optional`) for `GraphState` and Pydantic models |
| `os` | Reads and sets LangSmith tracing environment variables at startup |
| `sys` | Adjusts `sys.path` for direct script execution of `e2e_graph.py` |

---

## Optional: LangSmith Observability

LangSmith tracing is activated automatically when `LANGCHAIN_API_KEY` is present in the environment. It is provided by the `langsmith` package, which is a transitive dependency of `langchain-core` and does not need to be installed separately. When active, it captures full LLM request/response payloads, token usage, latency, and StateGraph node execution traces, viewable at [smith.langchain.com](https://smith.langchain.com) under the **RCSA** project.
