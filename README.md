# CodeReview AI 🔍

> An autonomous code review agent that analyzes pull requests, classifies issue severity using a fine-tuned model, and orchestrates a multi-step review workflow with LangGraph.

![Architecture](docs/architecture.png)

---

## What It Does

CodeReview AI reviews a git diff like a senior engineer would:
1. **Parses** the diff into structured hunks (files, functions, imports changed)
2. **Runs an AI agent** that autonomously decides which tools to call
3. **Classifies severity** using a fine-tuned Phi-2 model (`critical / warning / suggestion`)
4. **Routes the output** — critical issues trigger a merge block recommendation

---

## Architecture

```
PR Diff → ingest_pr → agent_tool_loop → classify_severity → [branch] → format_output
                            ↓
                     Tools available:
                     • static_analysis_tool  (pylint-based bug detection)
                     • test_coverage_tool    (checks for missing tests)
                     • docs_fetch_tool       (library safety knowledge base)
```

The pipeline is built with **LangGraph** — a stateful graph framework that supports
conditional branching and agent tool-call cycles.

---

## Fine-Tuned Severity Classifier

The `classify_severity` node uses a **Phi-2 model fine-tuned with LoRA** on 120 labeled
code issue descriptions.

| Class | F1 Score |
|---|---|
| `critical` | 0.857 |
| `warning` | 0.667 |
| `suggestion` | 0.800 |
| **weighted avg** | **0.767** |

Training details:
- Base model: `microsoft/phi-2` (2.7B parameters)
- Method: LoRA (r=8, alpha=16) via `peft`
- Trainable parameters: 2.6M out of 2.65B (0.099%)
- Epochs: 5 | Dataset: 120 labeled examples (balanced across 3 classes)
- Training notebook: `training/train.py`

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `langgraph` | Stateful graph orchestration with conditional branching |
| `langchain-ollama` | Local LLM agent (qwen2.5) for tool-calling |
| `transformers` + `peft` | Fine-tuning Phi-2 with LoRA |
| `pylint` / `ast` | Static analysis inside the review tool |
| `datasets` | Dataset preparation for fine-tuning |
| `Ollama` | Local LLM inference — no API cost |

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/YOUR_USERNAME/code-review-ai
cd code-review-ai
python3.11 -m venv venv && source venv/bin/activate
pip install langgraph langchain-core langchain-community langchain-ollama \
            transformers peft datasets accelerate torch pylint

# 2. Start Ollama
ollama pull qwen2.5
ollama serve

# 3. Run demo
python review.py --demo

# 4. Review a real diff
git diff main your-branch | python review.py --stdin

# 5. Review a diff file
python review.py --diff path/to/changes.diff
```

---

## Sample Output

```
[NODE] ingest_pr — parsing PR diff into structured hunks...
[NODE] agent_tool_loop — agent reasoning over diff with tools...
  → Calling tool: static_analysis_tool(...)
  → Calling tool: test_coverage_tool(...)
[NODE] classify_severity — running fine-tuned model...
  → 'Undefined variable db...' → critical
[NODE] block_merge_recommendation — critical issue detected...
[NODE] format_output — assembling final review...

===== CODE REVIEW REPORT =====
⛔ MERGE BLOCKED: Critical issues must be resolved first.
🔴 [CRITICAL] Undefined variable 'db' (undefined-variable)
   Source: static_analysis_tool
🟡 [WARNING] No tests found for function 'process'
   Source: test_coverage_tool
```

---

## Project Structure

```
code-review-ai/
├── state.py              # ReviewState TypedDict — shared graph memory
├── nodes.py              # All 5 graph node functions
├── graph.py              # LangGraph assembly and conditional edges
├── tools.py              # 3 callable tools for the agent
├── classifier.py         # Fine-tuned Phi-2 severity classifier
├── review.py             # CLI entrypoint
├── training/
│   ├── severity_dataset.py   # 120 labeled training examples
│   ├── train.py              # LoRA fine-tuning script
│   └── severity_model/       # Saved LoRA adapter weights
├── sample_reviews/           # 3 example diffs + their outputs
└── docs/
    └── architecture.png      # LangGraph pipeline diagram
```

---

## What I Learned

- How to design a **stateful LangGraph pipeline** with cycles and conditional edges
- How **tool-calling agents** decide which tools to invoke based on context
- How to **fine-tune a large language model** with LoRA using a small custom dataset
- How to **integrate a fine-tuned model** as a callable component inside a larger AI pipeline
- The difference between keyword-based classification and learned semantic classification