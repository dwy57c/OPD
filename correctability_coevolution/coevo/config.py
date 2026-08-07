from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ModelEndpoint:
    model: str
    base_url: str
    api_key: str = "EMPTY"
    temperature: float = 0.2
    max_tokens: int = 256
    enable_thinking: bool = False
    seed: int | None = None

    @property
    def litellm_model(self) -> str:
        return f"hosted_vllm/{self.model}"

    @property
    def litellm_args(self) -> dict:
        api_base = self.base_url.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base += "/v1"
        args = {
            "api_base": api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            },
        }
        if self.seed is not None:
            args["seed"] = self.seed
        return args


@dataclass(frozen=True)
class HintEndpoint:
    model: str
    base_url: str
    api_key: str
    max_tokens: int = 6144
    timeout: float = 300.0
    retries: int = 3

    @classmethod
    def from_env(cls, required: bool = False) -> "HintEndpoint | None":
        base_url = os.getenv(
            "COEVO_TEACHER_HINT_URL",
            os.getenv("GOOGLE_BASE_URL", ""),
        ).rstrip("/")
        api_key = os.getenv(
            "COEVO_TEACHER_HINT_API_KEY",
            os.getenv("GOOGLE_API_KEY", ""),
        )
        if not base_url:
            if required:
                raise ValueError(
                    "COEVO_TEACHER_HINT_URL or GOOGLE_BASE_URL is required "
                    "for closed-model Teacher hints"
                )
            return None
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        return cls(
            model=os.getenv(
                "COEVO_TEACHER_HINT_MODEL",
                "gemini-3.1-pro-preview",
            ),
            base_url=base_url,
            api_key=api_key,
            max_tokens=int(
                os.getenv(
                    "COEVO_TEACHER_HINT_MAX_TOKENS",
                    "6144",
                )
            ),
            timeout=float(
                os.getenv(
                    "COEVO_TEACHER_HINT_TIMEOUT",
                    "300",
                )
            ),
            retries=int(
                os.getenv(
                    "COEVO_TEACHER_HINT_RETRIES",
                    "3",
                )
            ),
        )


@dataclass(frozen=True)
class InfraConfig:
    # Student and Teacher are two information views over this one policy model.
    # There is deliberately no separate Teacher model endpoint.
    policy: ModelEndpoint
    buyer_reference: ModelEndpoint
    nl_judge: ModelEndpoint | None = None
    domain: str = "airline"
    task_split: str = "train"
    task_id: str = "1"
    branch_max_steps: int = 24
    branch_max_tokens: int = 256
    # 0 means every natural Student action in the completed dialogue.
    max_intervention_decisions: int = 0
    buyer_plan_mode: str = "structured"
    continuations: int = 1
    nl_judge_max_tokens: int = 1024
    nl_judge_retries: int = 3
    seed: int = 42
    teacher_hint_mode: str = "closed_model"
    teacher_hinter: HintEndpoint | None = None

    def __post_init__(self):
        if self.teacher_hint_mode not in {"none", "closed_model"}:
            raise ValueError(
                "teacher_hint_mode must be 'none' or 'closed_model', got "
                f"{self.teacher_hint_mode!r}"
            )
        if self.teacher_hint_mode == "closed_model" and self.teacher_hinter is None:
            raise ValueError(
                "teacher_hinter is required when teacher_hint_mode='closed_model'"
            )
        if self.max_intervention_decisions < 0:
            raise ValueError(
                "max_intervention_decisions must be non-negative (0 means unlimited)"
            )
        if self.buyer_plan_mode not in {"structured", "legacy"}:
            raise ValueError(
                "buyer_plan_mode must be 'structured' or 'legacy', got "
                f"{self.buyer_plan_mode!r}"
            )
        if self.nl_judge_retries < 1:
            raise ValueError("nl_judge_retries must be positive")

    @property
    def student(self) -> ModelEndpoint:
        """Compatibility/readability alias: Student is the shared policy."""
        return self.policy

    @property
    def teacher(self) -> ModelEndpoint:
        """Compatibility/readability alias: Teacher is the same shared policy."""
        return self.policy

    @classmethod
    def from_env(cls) -> "InfraConfig":
        policy_port = os.getenv("COEVO_POLICY_PORT", "8000")
        buyer_port = os.getenv("COEVO_BUYER_PORT", "8002")
        policy_model = os.getenv("COEVO_POLICY_MODEL", "Qwen3-4B")
        policy_url = os.getenv(
            "COEVO_POLICY_URL",
            f"http://127.0.0.1:{policy_port}",
        )
        hint_mode = os.getenv("COEVO_TEACHER_HINT_MODE", "closed_model")
        return cls(
            policy=ModelEndpoint(policy_model, policy_url),
            buyer_reference=ModelEndpoint(
                os.getenv("COEVO_BUYER_MODEL", "Qwen3-4B"),
                os.getenv("COEVO_BUYER_URL", f"http://127.0.0.1:{buyer_port}"),
            ),
            nl_judge=ModelEndpoint(
                os.getenv("COEVO_NL_JUDGE_MODEL", policy_model),
                os.getenv("COEVO_NL_JUDGE_URL", policy_url),
                os.getenv("COEVO_NL_JUDGE_API_KEY", "EMPTY"),
            ),
            domain=os.getenv("COEVO_DOMAIN", "airline"),
            task_split=os.getenv("COEVO_TASK_SPLIT", "train"),
            task_id=os.getenv("COEVO_TASK_ID", "1"),
            branch_max_steps=int(os.getenv("COEVO_BRANCH_MAX_STEPS", "24")),
            branch_max_tokens=int(os.getenv("COEVO_BRANCH_MAX_TOKENS", "256")),
            max_intervention_decisions=int(
                os.getenv("COEVO_MAX_INTERVENTION_DECISIONS", "0")
            ),
            buyer_plan_mode=os.getenv("COEVO_BUYER_PLAN_MODE", "structured"),
            continuations=int(os.getenv("COEVO_CONTINUATIONS", "1")),
            nl_judge_max_tokens=int(os.getenv("COEVO_NL_JUDGE_MAX_TOKENS", "1024")),
            nl_judge_retries=int(os.getenv("COEVO_NL_JUDGE_RETRIES", "3")),
            seed=int(os.getenv("COEVO_SEED", "42")),
            teacher_hint_mode=hint_mode,
            teacher_hinter=HintEndpoint.from_env(required=hint_mode == "closed_model"),
        )
