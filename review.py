# CLI entrypoint

from graph import build_graph

SAMPLE_DIFF = """
--- a/app.py
+++ b/app.py
@@ -10,6 +10,12 @@
 def ger_user(user_id):
+   result = db.query(user_id)
+   return result.data
+
+def process(input):
+   x = int(input)
+   return x / 0 
"""

def main():
    app = build_graph()
    initial_state = {
        "pr_diff": SAMPLE_DIFF,
        "issues_found": [],
        "severity_labels": [],
        "final_output": "",
        "route": "",
    }

    print("Starting CodeReview AI pipeline...\n")
    final_state = app.invoke(initial_state)
    print("\n===== PIPELINE COMPLETE =====")

if __name__ == "__main__":
    main()