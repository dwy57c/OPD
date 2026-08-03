from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ModelEndpoint:
    model: str
    base_url: str

    @property
    def litellm_model(self) -> str:
        return f"hosted_vllm/{self.model}"

    @property
    def litellm_args(self) -> dict:
        return {
            "api_base": self.base_url.rstrip("/") + "/v1",
            "api_key": "EMPTY",
            "temperature": 0.2,
            "max_tokens": 256,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }


@dataclass(frozen=True)
class InfraConfig:
    teacher: ModelEndpoint
    student: ModelEndpoint
    buyer_reference: ModelEndpoint
    domain: str = "airline"
    task_id: str = "1"
    branch_max_steps: int = 24
    branch_max_tokens: int = 256
    cutoffs_per_turn: int = 2
    max_cutoff_turns: int = 3
    continuations: int = 1
    correctability_prior: float = 1.0
    seed: int = 42

    @classmethod
    def from_env(cls) -> "InfraConfig":
        return cls(
            teacher=ModelEndpoint(
                os.getenv("COEVO_TEACHER_MODEL", "Qwen3-32B"),
                os.getenv("COEVO_TEACHER_URL", "http://127.0.0.1:8000"),
            ),
            student=ModelEndpoint(
                os.getenv("COEVO_STUDENT_MODEL", "Qwen3-4B"),
                os.getenv("COEVO_STUDENT_URL", "http://127.0.0.1:8001"),
            ),
            buyer_reference=ModelEndpoint(
                os.getenv("COEVO_BUYER_MODEL", "Qwen3-4B"),
                os.getenv("COEVO_BUYER_URL", "http://127.0.0.1:8002"),
            ),
            domain=os.getenv("COEVO_DOMAIN", "airline"),
            task_id=os.getenv("COEVO_TASK_ID", "1"),
            branch_max_steps=int(os.getenv("COEVO_BRANCH_MAX_STEPS", "24")),
            branch_max_tokens=int(os.getenv("COEVO_BRANCH_MAX_TOKENS", "256")),
            cutoffs_per_turn=int(os.getenv("COEVO_CUTOFFS_PER_TURN", "2")),
            max_cutoff_turns=int(os.getenv("COEVO_MAX_CUTOFF_TURNS", "3")),
            continuations=int(os.getenv("COEVO_CONTINUATIONS", "1")),
            correctability_prior=float(os.getenv("COEVO_CORRECTABILITY_PRIOR", "1.0")),
            seed=int(os.getenv("COEVO_SEED", "42")),
        )
