# tools.py
import ast
import subprocess
from langchain_core.tools import tool

@tool
def static_analysis_tool(code_snippet: str) -> str:
    """
    Analyzes a Python code snippet for syntax errors, undefined variables,
    security vulnerabilities, and common bugs. Use this whenever the diff
    contains Python code changes.
    Returns a list of issues found, or 'No issues found' if the code is clean.
    """
    issues = []

    # Check 1: AST parse for syntax errors
    try:
        ast.parse(code_snippet)
    except SyntaxError as e:
        issues.append(f"Syntax error at line {e.lineno}: {e.msg}")

    # Check 2: Security pattern detection  ← NEW BLOCK
    security_patterns = [
        ("shell=True", "Shell injection risk: subprocess called with shell=True and user input"),
        ("password", "Hardcoded credential detected: variable named 'password' found in code"),
        ("secret", "Hardcoded credential detected: variable named 'secret' found in code"),
        ("api_key", "Hardcoded credential detected: variable named 'api_key' found in code"),
        ("eval(", "Code injection risk: eval() executes arbitrary code"),
        ("exec(", "Code injection risk: exec() executes arbitrary code"),
        ("pickle.loads", "Deserialization risk: pickle.loads can execute arbitrary code"),
        ("md5(", "Weak cryptography: MD5 is cryptographically broken"),
        ("sha1(", "Weak cryptography: SHA1 is cryptographically broken"),
        ("except:", "Bare except clause silences all errors including system exits"),
        ("except Exception:", "Overly broad exception handling hides real errors"),
    ]

    lines = code_snippet.splitlines()
    for i, line in enumerate(lines, 1):
        for pattern, message in security_patterns:
            if pattern in line:
                issues.append(f"line {i}: {message}")

    # Check 3: Run pylint for undefined variables and errors
    try:
        with open("/tmp/_review_snippet.py", "w") as f:
            f.write(code_snippet)

        result = subprocess.run(
            ["pylint", "/tmp/_review_snippet.py",
             "--disable=all",
             "--enable=E",
             "--output-format=text",
             "--score=no"],
            capture_output=True, text=True, timeout=10
        )
        pylint_output = result.stdout.strip()
        meaningful_lines = [
            line.replace("/tmp/_review_snippet.py", "reviewed file")
            for line in pylint_output.splitlines()
            if line.strip()
            and not line.startswith("***")
            and not line.startswith("---")
            and not line.startswith("Your code has been rated")
            and not line.startswith("------")
        ]
        if meaningful_lines:
            issues.append("\n".join(meaningful_lines))

    except Exception as e:
        issues.append(f"pylint unavailable: {str(e)}")

    return "\n".join(issues) if issues else "No issues found"

@tool
def test_coverage_tool(function_name: str) -> str:
    """
    Checks whether a given function name has any corresponding test in the codebase. Use this when the diff  adds or modifies a function definition. Returns whether test exist and their names if found.
    """
    import os
    import glob

    # Collect test files but explicitly exclude venv and hidden dirs
    all_py_files = glob.glob("**/*.py", recursive=True)
    test_files = [
        f for f in all_py_files
        if ("test" in os.path.basename(f).lower())
        and not f.startswith("venv/")
        and not f.startswith(".venv/")
        and "site-packages" not in f
    ]

    if not test_files:
        return f"No test files found in project. Function '{function_name}' is untested."
    
    matches = []
    for filepath in test_files:
        try:
            with open(filepath, "r") as f:
                content = f.read()
            if function_name in content:
                matches.append(filepath)
        except Exception:
            continue

    if matches:
        return f"Test found for '{function_name}' in: {','.join(matches)}"
    else:
        return f"No tests found for function '{function_name}'. Consider adding test coverage."
    
@tool
def docs_fetch_tool(library_name: str) -> str:
    """
    Fetches a brief description of a Python library or built-in module.
    Use this when the diff imports an unfamiliar library or uses an API
    you want to verify is being used correctly.
    Returns a short summary of what the library does.
    """
    # Curated local knowledge base — no API calls needed
    known_libs = {
        "os": "Standard library for OS interaction: file paths, env vars, process management.",
        "sys": "Access to Python interpreter internals: argv, path, exit codes.",
        "ast": "Parse and analyze Python source code as abstract syntax trees.",
        "subprocess": "Spawn and manage subprocesses; use with caution — shell injection risk if inputs are unsanitized.",
        "requests": "HTTP library for API calls. Watch for missing timeout parameters and unhandled status codes.",
        "flask": "Lightweight web framework. Common issues: debug=True in production, no input validation.",
        "django": "Full-stack web framework. Watch for raw SQL queries bypassing ORM, missing CSRF protection.",
        "sqlalchemy": "ORM for database access. Prefer parameterized queries; avoid string formatting in queries.",
        "pandas": "Data manipulation library. Watch for inplace operations and chained indexing warnings.",
        "numpy": "Numerical computing. Common issue: silent integer overflow in certain dtypes.",
        "pytest": "Testing framework. Ensure fixtures are properly scoped and teardown is handled.",
        "json": "Standard library JSON encoder/decoder. Watch for unhandled JSONDecodeError.",
        "re": "Regular expressions. Complex patterns can cause catastrophic backtracking.",
        "smtplib": "Email sending library. Never hardcode credentials; always use environment variables for passwords.",
    }

    name = library_name.lower().strip()
    if name in known_libs:
        return f"{library_name}: {known_libs[name]}"
    else:
        return f"'{library_name}' not in local knowledge base. Verify its usage manually or check official docs."