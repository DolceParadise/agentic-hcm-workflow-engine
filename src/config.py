from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    openrouter_api_key: str
    openrouter_model: str = "openai/gpt-oss-120b:free"
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    employee_id: str = "E001"
    db_path: Path = Path(".runtime/hcm.db")
    index_path: Path = Path(".runtime/policy_index.npz")
    policy_path: Path = Path("data/hr_policy_corpus.txt")
    employee_data_path: Path = Path("data/tech_company_employee_data_1000_with_leave.csv")
    compliance_rules_path: Path = Path("data/compliance_rules.json")
    retrieval_top_k: int = 4
    retrieval_threshold: float = 0.28
    auto_action_threshold: float = 0.90

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> Settings:
        root = project_root or Path(__file__).resolve().parents[1]

        def rooted(name: str, default: str) -> Path:
            value = Path(os.getenv(name, default))
            return value if value.is_absolute() else root / value

        return cls(
            project_root=root,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"),
            employee_id=os.getenv("HCM_EMPLOYEE_ID", "E001"),
            db_path=rooted("HCM_DB_PATH", ".runtime/hcm.db"),
            index_path=rooted("HCM_INDEX_PATH", ".runtime/policy_index.npz"),
            policy_path=rooted("HCM_POLICY_PATH", "data/hr_policy_corpus.txt"),
            employee_data_path=rooted(
                "HCM_EMPLOYEE_DATA_PATH",
                "data/tech_company_employee_data_1000_with_leave.csv",
            ),
            compliance_rules_path=rooted("HCM_COMPLIANCE_RULES_PATH", "data/compliance_rules.json"),
            auto_action_threshold=float(os.getenv("HCM_AUTO_ACTION_THRESHOLD", "0.90")),
        )
