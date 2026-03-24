## AGENT STRUCTURE 
┌─────────────────────────────────┐
                    │         Batch Orchestrator       │
                    │  (your script, not the graph)    │
                    └────────────┬────────────────────┘
                                 │  iterates 1000 records
                    ┌────────────▼────────────────────┐
                    │         StateGraph               │
                    │  ┌────┐  ┌────┐  ┌────┐        │
                    │  │ n1 │→ │ n2 │→ │ n3 │        │  × 1000
                    │  └────┘  └────┘  └────┘        │
                    │  returns: RecordResult           │
                    └────────────┬────────────────────┘
                                 │  collect results
                    ┌────────────▼────────────────────┐
                    │      Post-Graph Aggregator       │
                    │  join → validate → write file    │
                    └─────────────────────────────────┘
                    
## PROJECT STRUCTURE
rcsa_qc/
│
├── prompts/
│   ├── registry.yaml                    # master prompt registry
│   │
│   ├── audit/
│   │   └── rcsa_control_qc.yaml         # the prompt we just built
│   │
│   ├── aml/
│   │   └── aml_sar_narrative.yaml
│   │
│   └── ops_risk/
│       └── rcsa_reviewer.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── loader.py                    # PromptLoader class (step 2)
│   │   └── registry.py                 # PromptRegistry class (step 4)
│   │
│   ├── chains/
│   │   ├── __init__.py
│   │   └── rcsa_qc_chain.py            # chain wiring (step 3)
│   │
│   └── utils/
│       ├── __init__.py
│       └── logging.py                  # inference logging w/ prompt metadata
│
├── tests/
│   ├── prompts/
│   │   └── test_loader.py              # validate YAML schema + template vars
│   └── chains/
│       └── test_rcsa_qc_chain.py
│
├── notebooks/
│   └── rcsa_qc_dev.ipynb               # prompt iteration sandbox
│
├── configs/
│   └── settings.yaml                   # env-level config (model defaults, etc.)
│
├── .env                                 # ANTHROPIC_API_KEY (never committed)
├── .gitignore
├── requirements.txt
└── README.md