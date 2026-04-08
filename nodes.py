# All graph node functions

from state import ReviewState
from tools import static_analysis_tool, test_coverage_tool, docs_fetch_tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import re

# ── helper: parse raw diff into structured hunks ──────────────────────────────

def parse_diff(diff_text: str) -> list[dict]:
    """
    Breaks a raw git diff into a list of hunks, each with:
    - filename, added_lines (code), functions_changed, imports_added
    """
    hunks = []
    current_file = None
    added_lines = []

    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            if current_file and added_lines:
                hunks.append(_build_hunk(current_file, added_lines))
            current_file = line[6:].strip()
            added_lines = []
        elif line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:]) # strip the leading +

    if current_file and added_lines:
        hunks.append(_build_hunk(current_file, added_lines))

    return hunks

def _build_hunk(filename: str, addded_lines: list[str]) -> dict:
    code = "\n".join(addded_lines)
    functions = re.findall(r"def (\w+)\s*\(", code)
    imports = re.findall(r"import (\w+)",code)
    return {
        "filename":filename,
        "added_code": code,
        "functions_changed": functions,
        "imports_added": imports,
    }

# ── node 1: ingest_pr ─────────────────────────────────────────────────────────

def ingest_pr(state: ReviewState) -> dict:
    print("\n[NODE] ingest_pr - parsing PR diff into structured hunks...")
    hunks = parse_diff(state["pr_diff"])
    for h in hunks:
        print(f" File: {h['filename']}")
        print(f" Functions changed: {h['functions_changed']}")
        print(f" Imports added: {h['imports_added']}")
    return {"parsed_hunks": hunks}

# ── node 2: agent_tool_loop ───────────────────────────────────────────────────

def agent_tool_loop(state: ReviewState) -> dict:
    print("\n[NODE] agent_tool_loop - agent reasoning over diff with tools...")

    llm = ChatOllama(
        model = "qwen2.5:latest",
        temperature=0,
        num_ctx=4096,
        num_predict=1024,
    )

    tools = [static_analysis_tool, test_coverage_tool, docs_fetch_tool]
    agent = llm.bind_tools(tools)

    # Build a focused context from parsed hunks
    hunk_summary = "" 
    for h in state["parsed_hunks"]:
        hunk_summary += f"\nFile: {h['filename']}\n"
        hunk_summary += f"Added code:\n{h['added_code']}\n"
        hunk_summary += f"Functions defined: {h['functions_changed']}\n"
        hunk_summary += f"Imports used: {h['imports_added']}\n"

    messages = [
        SystemMessage(content="""You are a senior code reviewer. 
                      You have three tools available:
                      - static_analysis_tool: use on any Python code snippet to find bugs and errors
                      - test_coverage_tool: use on any function name to check if it has tests
                      - docs_fetch_tool: use on any library name to check if it's being used safely

                      Review the diff below. Call the appropriate tools. Be selective - only call a tool when it will
                      reveal useful information. After calling tools, summarize what issues you found.
                      """),
                      HumanMessage(content=f"Review this diff:\n{hunk_summary}")
    ]

    all_issues = []
    max_iterations = 5 # prevent infinite loops

    for i in range(max_iterations):
        print(f" Agent iterations {i+1}...")
        response = agent.invoke(messages)

        # If no tool calls, agent is done reasoning
        if not response.tool_calls:
            print(" Agent finished reasoning.")
            break

        # Execute each tool the agent called
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f" -> Calling tool: {tool_name}({tool_args}) ")

            # Route to the correct tool function
            tool_map = {
                "static_analysis_tool": static_analysis_tool,
                "test_coverage_tool": test_coverage_tool,
                "docs_fetch_tool": docs_fetch_tool
            }

            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
                print(f" Result: {result[:100]}...") # print first 100 chars

                # Convert tool result into an Issue if it found something real
                if "No issues found" not in result and "Tests found" not in result:
                    all_issues.append({
                        "description": result[:200],
                        "location": f"detected by {tool_name}",
                        "severity": None,
                        "source_tool": tool_name,
                    })
                
                # Feed the tool result back into the conversation
                from langchain_core.messages import ToolMessage
                messages.append(response)
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id = tool_call["id"]
                ))
    
    print(f" Total issues found: {len(all_issues)}")
    return {"issues_found": all_issues}

def classify_severity(state: ReviewState) -> dict:
    print("\n[NODE] classify_severity — labeling issue severity...")
    labels = []

    for issue in state["issues_found"]:
        desc = issue["description"].lower()
        tool = issue["source_tool"]
        label = "suggestion"  # default

        if tool == "static_analysis_tool":
            # pylint error codes: E = error (critical), W = warning
            if " e" in desc or "error" in desc or "division" in desc or "syntax" in desc:
                label = "critical"
            elif " w" in desc or "warning" in desc:
                label = "warning"

        elif tool == "test_coverage_tool":
            if "untested" in desc or "no tests found" in desc:
                label = "warning"
            else:
                label = "suggestion"

        elif tool == "docs_fetch_tool":
            if "injection" in desc or "caution" in desc or "avoid" in desc:
                label = "warning"
            else:
                label = "suggestion"

        labels.append(label)

    print(f"  Labels: {labels}")
    route = "critical" if "critical" in labels else "standard"
    return {"severity_labels": labels, "route": route}

def block_merge_recommendation(state: ReviewState) -> dict:
    print("\n[NODE] block_merge_recommendation - critical issue found, recommending block...")
    return {"final_output": state.get("final_output", "") + "\n⛔ MERGE BLOCKED: Critical issues must be resolved first."}

def format_output(state: ReviewState) -> dict: 
    print("\n[NODE] format_output — assembling final review...")
    issues = state.get("issues_found",[])
    labels = state.get("severity_labels",[])
    prefix = state.get("final_output", "")

    lines = ["==== CODE REVIEW REPORT ===="]
    if prefix:
        lines.append(prefix)

    if not issues:
        lines.append("✅ No issues found. Looks good to merge.")
    else:
        for issue, label in zip(issues, labels):
            emoji = {"critical": "🔴", "warning": "🟡", "suggestion": "🟢"}.get(label, "⚪")
            lines.append(f"{emoji} [{label.upper()}] {issue['description'][:120]}")
            lines.append(f"   Source: {issue['source_tool']}")

    output = "\n".join(lines)
    print(output)
    return {"final_output": output}
