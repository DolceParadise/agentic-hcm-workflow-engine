from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from hcm_engine.engine import WorkflowEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HCM workflow engine.")
    parser.add_argument("message")
    parser.add_argument("--session-id")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    result = WorkflowEngine().run(args.message, args.session_id)
    print(result.response)
    print(f"session_id={result.session_id}")
    if args.trace:
        print(json.dumps([step.to_dict() for step in result.trace], indent=2))
        print(json.dumps({"tokens": result.token_usage, "cost": result.cost}, indent=2))

