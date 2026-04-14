# review.py
import argparse
import sys
from graph import build_graph

# ── Sample diff for demo mode ─────────────────────────────────────────────────
SAMPLE_DIFF = """
--- a/app.py
+++ b/app.py
@@ -10,6 +10,12 @@
 def get_user(user_id):
+    result = db.query(user_id)
+    return result.data
+
+def process(input):
+    x = int(input)
+    return x / 0
"""

def run_review(diff_text: str):
    app = build_graph()
    initial_state = {
        "pr_diff": diff_text,
        "parsed_hunks": [],
        "issues_found": [],
        "severity_labels": [],
        "final_output": "",
        "route": "",
    }
    print("Starting CodeReview AI pipeline...\n")
    final_state = app.invoke(initial_state)
    print("\n===== PIPELINE COMPLETE =====")
    return final_state["final_output"]


def main():
    parser = argparse.ArgumentParser(
        description="CodeReview AI — Automated PR reviewer powered by LangGraph + fine-tuned Phi-2"
    )
    parser.add_argument(
        "--diff",
        type=str,
        help="Path to a .diff file to review",
        default=None,
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read diff from stdin (pipe from git diff)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in sample diff",
    )
    args = parser.parse_args()

    # Determine input source
    if args.stdin:
        print("[Input] Reading diff from stdin...")
        diff_text = sys.stdin.read()
    elif args.diff:
        print(f"[Input] Reading diff from file: {args.diff}")
        with open(args.diff, "r") as f:
            diff_text = f.read()
    else:
        # Default to demo mode
        print("[Input] No input specified — running in demo mode.")
        print("[Input] Usage: python review.py --diff file.diff")
        print("[Input] Usage: git diff main | python review.py --stdin\n")
        diff_text = SAMPLE_DIFF

    run_review(diff_text)


if __name__ == "__main__":
    main()