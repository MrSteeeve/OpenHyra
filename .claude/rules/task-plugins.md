New tasks live in tasks/<name>/ with task.json, TASK.md, evaluator.py, and optionally seed_solution/.
task.json schema: name, protocol, direction, editable_files are required; see bermudan_optimal_stopping/task.json for reference.
evaluator.py is trusted code that runs outside the sandbox — it must never import or exec candidate code.
Formalization modules are optional; they follow the verify_formalization_request interface.
