from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ModelEndpoint:
    model: str
    base_url: str
    api_key: str = "EMPTY"

    @property
    def litellm_model(self) -> str:
        return f"hosted_vllm/{self.model}"

    @property
    def litellm_args(self) -> dict:
        return {
            "api_base": self.base_url.rstrip("/") + "/v1",
            "api_key": self.api_key,
            "temperature": 0.2,
            "max_tokens": 256,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }


@dataclass(frozen=True)
class InfraConfig:
    teacher: ModelEndpoint
    student: ModelEndpoint
    buyer_reference: ModelEndpoint
    nl_judge: ModelEndpoint | None = None
    domain: str = "airline"
    task_split: str = "train"
    task_id: str = "1"
    branch_max_steps: int = 24
    branch_max_tokens: int = 256
    cutoffs_per_turn: int = 2
    max_cutoff_turns: int = 3
    continuations: int = 1
    correctability_prior: float = 1.0
    nl_judge_max_tokens: int = 1024
    seed: int = 42

    @classmethod
    def from_env(cls) -> "InfraConfig":
        teacher_port = os.getenv("COEVO_TEACHER_PORT", "8000")
        student_port = os.getenv("COEVO_STUDENT_PORT", "8001")
        buyer_port = os.getenv("COEVO_BUYER_PORT", "8002")
        teacher_model = os.getenv("COEVO_TEACHER_MODEL", "Qwen3-32B")
        teacher_url = os.getenv("COEVO_TEACHER_URL", f"http://127.0.0.1:{teacher_port}")
        return cls(
            teacher=ModelEndpoint(teacher_model, teacher_url),
            student=ModelEndpoint(
                os.getenv("COEVO_STUDENT_MODEL", "Qwen3-4B"),
                os.getenv("COEVO_STUDENT_URL", f"http://127.0.0.1:{student_port}"),
            ),
            buyer_reference=ModelEndpoint(
                os.getenv("COEVO_BUYER_MODEL", "Qwen3-4B"),
                os.getenv("COEVO_BUYER_URL", f"http://127.0.0.1:{buyer_port}"),
            ),
            nl_judge=ModelEndpoint(
                os.getenv("COEVO_NL_JUDGE_MODEL", teacher_model),
                os.getenv("COEVO_NL_JUDGE_URL", teacher_url),
                os.getenv("COEVO_NL_JUDGE_API_KEY", "EMPTY"),
            ),
            domain=os.getenv("COEVO_DOMAIN", "airline"),
            task_split=os.getenv("COEVO_TASK_SPLIT", "train"),
            task_id=os.getenv("COEVO_TASK_ID", "1"),
            branch_max_steps=int(os.getenv("COEVO_BRANCH_MAX_STEPS", "24")),
            branch_max_tokens=int(os.getenv("COEVO_BRANCH_MAX_TOKENS", "256")),
            cutoffs_per_turn=int(os.getenv("COEVO_CUTOFFS_PER_TURN", "2")),
            max_cutoff_turns=int(os.getenv("COEVO_MAX_CUTOFF_TURNS", "3")),
            continuations=int(os.getenv("COEVO_CONTINUATIONS", "1")),
            correctability_prior=float(os.getenv("COEVO_CORRECTABILITY_PRIOR", "1.0")),
            nl_judge_max_tokens=int(os.getenv("COEVO_NL_JUDGE_MAX_TOKENS", "1024")),
            seed=int(os.getenv("COEVO_SEED", "42")),
        )
