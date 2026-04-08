# ReviewState definition

from typing import TypedDict, List, Optional

class Issue(TypedDict):
    description: str
    location: str # e.g. "line 42" or "function get_user()"
    severity: Optional[str] # will be filled in Phase 3: critical/warning/suggestion
    source_tool: str # "static_analysis" | "test_coverage" | "docs_fetch"

class ReviewState(TypedDict):
    pr_diff:str                 # raw input: the code diff text
    parsed_hunks : List[dict]   # structured diff parsed by ingest_pr
    issues_found: List[Issue]   # accumulated by the agent in Phase 2
    severity_labels: List[str]  # filled by classifier in Phase 3
    final_output: str           # formatted review report
    route: str                  # controls branching: "critical"| "standard"


