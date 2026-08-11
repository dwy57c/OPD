from dataclasses import dataclass
import os

from coevo.artifacts import model_manifest_revision


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
    previous_policy: ModelEndpoint | None = None
    nl_judge: ModelEndpoint | None = None
    domain: str = "airline"
    task_split: str = "train"
    task_id: str = "1"
    branch_max_steps: int = 24
    branch_max_tokens: int = 256
    # 0 means every natural Student action in the completed dialogue.
    max_teacher_targets: int = 0
    buyer_plan_mode: str = "structured"
    teacher_validation_continuations: int = 1
    nl_judge_max_tokens: int = 1024
    nl_judge_retries: int = 3
    seed: int = 42
    teacher_hint_mode: str = "closed_model"
    teacher_hinter: HintEndpoint | None = None
    current_policy_checkpoint: str = ""
    previous_policy_checkpoint: str = ""
    buyer_checkpoint: str = ""
    round_index: int = 0
    dataset_schema_version: int = 4
    target_schema_version: int = 2
    teacher_target_version: str = "skill-contrast-sharpened-v2"
    tokenizer_id: str = ""
    current_policy_revision: str = ""
    previous_policy_revision: str = ""
    buyer_revision: str = ""
    teacher_gap_topk: int = 20
    teacher_gap_min_support_mass: float = 0.95
    teacher_gap_eps: float = 1e-8
    skill_gate_metric: str = "forward_kl"
    skill_gate_eps: float = 1e-8
    skill_gate_low: float = 0.0
    skill_gate_high: float = 0.05
    skill_sharpen_t_min: float = 0.7

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
        if self.max_teacher_targets < 0:
            raise ValueError(
                "max_teacher_targets must be non-negative (0 means unlimited)"
            )
        if self.buyer_plan_mode not in {"structured", "legacy"}:
            raise ValueError(
                "buyer_plan_mode must be 'structured' or 'legacy', got "
                f"{self.buyer_plan_mode!r}"
            )
        if self.nl_judge_retries < 1:
            raise ValueError("nl_judge_retries must be positive")
        if self.teacher_validation_continuations < 1:
            raise ValueError("teacher_validation_continuations must be positive")
        if self.dataset_schema_version != 4:
            raise ValueError("only dataset_schema_version=4 is supported")
        if self.target_schema_version != 2:
            raise ValueError("only target_schema_version=2 is supported")
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        if self.teacher_gap_topk < 1:
            raise ValueError("teacher_gap_topk must be positive")
        if not 0 < self.teacher_gap_min_support_mass <= 1:
            raise ValueError("teacher_gap_min_support_mass must be in (0, 1]")
        if self.teacher_gap_eps <= 0:
            raise ValueError("teacher_gap_eps must be positive")
        if self.skill_gate_metric != "forward_kl":
            raise ValueError("only forward_kl skill contrast is supported")
        if self.skill_gate_eps <= 0:
            raise ValueError("skill_gate_eps must be positive")
        if self.skill_gate_low < 0 or self.skill_gate_high <= self.skill_gate_low:
            raise ValueError("skill gate thresholds must satisfy 0 <= low < high")
        if not 0 < self.skill_sharpen_t_min < 1:
            raise ValueError("skill_sharpen_t_min must be in (0, 1)")

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
        previous_policy_port = os.getenv("COEVO_PREVIOUS_POLICY_PORT", "8001")
        buyer_port = os.getenv("COEVO_BUYER_PORT", "8002")
        policy_model = os.getenv("COEVO_POLICY_MODEL", "Qwen3-4B")
        policy_path = os.getenv("COEVO_POLICY_PATH", policy_model)
        buyer_path = os.getenv("COEVO_BUYER_PATH", policy_path)
        previous_policy_path = os.getenv("COEVO_PREVIOUS_POLICY_PATH", "")
        branch_max_tokens = int(os.getenv("COEVO_BRANCH_MAX_TOKENS", "256"))
        nl_judge_max_tokens = int(os.getenv("COEVO_NL_JUDGE_MAX_TOKENS", "1024"))
        current_checkpoint = os.getenv("COEVO_CURRENT_POLICY_CHECKPOINT", policy_path)
        previous_checkpoint = os.getenv(
            "COEVO_PREVIOUS_POLICY_CHECKPOINT", previous_policy_path
        )
        buyer_checkpoint = os.getenv("COEVO_BUYER_CHECKPOINT", buyer_path)
        policy_url = os.getenv(
            "COEVO_POLICY_URL",
            f"http://127.0.0.1:{policy_port}",
        )
        hint_mode = os.getenv("COEVO_TEACHER_HINT_MODE", "closed_model")
        previous_policy_url = os.getenv("COEVO_PREVIOUS_POLICY_URL", "")
        previous_policy = None
        if previous_policy_url or previous_policy_path:
            previous_policy = ModelEndpoint(
                os.getenv("COEVO_PREVIOUS_POLICY_MODEL", policy_model),
                previous_policy_url
                or f"http://127.0.0.1:{previous_policy_port}",
                max_tokens=branch_max_tokens,
            )
        return cls(
            policy=ModelEndpoint(
                policy_model,
                policy_url,
                max_tokens=branch_max_tokens,
            ),
            buyer_reference=ModelEndpoint(
                os.getenv("COEVO_BUYER_MODEL", "Qwen3-4B"),
                os.getenv("COEVO_BUYER_URL", f"http://127.0.0.1:{buyer_port}"),
                max_tokens=branch_max_tokens,
            ),
            previous_policy=previous_policy,
            nl_judge=ModelEndpoint(
                os.getenv("COEVO_NL_JUDGE_MODEL", policy_model),
                os.getenv("COEVO_NL_JUDGE_URL", policy_url),
                os.getenv("COEVO_NL_JUDGE_API_KEY", "EMPTY"),
                max_tokens=nl_judge_max_tokens,
            ),
            domain=os.getenv("COEVO_DOMAIN", "airline"),
            task_split=os.getenv("COEVO_TASK_SPLIT", "train"),
            task_id=os.getenv("COEVO_TASK_ID", "1"),
            branch_max_steps=int(os.getenv("COEVO_BRANCH_MAX_STEPS", "24")),
            branch_max_tokens=branch_max_tokens,
            max_teacher_targets=int(
                os.getenv("COEVO_MAX_TEACHER_TARGETS", "0")
            ),
            buyer_plan_mode=os.getenv("COEVO_BUYER_PLAN_MODE", "structured"),
            teacher_validation_continuations=int(
                os.getenv("COEVO_TEACHER_VALIDATION_CONTINUATIONS", "1")
            ),
            nl_judge_max_tokens=nl_judge_max_tokens,
            nl_judge_retries=int(os.getenv("COEVO_NL_JUDGE_RETRIES", "3")),
            seed=int(os.getenv("COEVO_SEED", "42")),
            teacher_hint_mode=hint_mode,
            teacher_hinter=HintEndpoint.from_env(required=hint_mode == "closed_model"),
            current_policy_checkpoint=current_checkpoint,
            previous_policy_checkpoint=previous_checkpoint,
            buyer_checkpoint=buyer_checkpoint,
            round_index=int(os.getenv("COEVO_ROUND_INDEX", "0")),
            dataset_schema_version=int(
                os.getenv("COEVO_DATASET_SCHEMA_VERSION", "4")
            ),
            target_schema_version=int(os.getenv("COEVO_TARGET_SCHEMA_VERSION", "2")),
            teacher_target_version=os.getenv(
                "COEVO_TEACHER_TARGET_VERSION", "skill-contrast-sharpened-v2"
            ),
            tokenizer_id=os.getenv(
                "COEVO_TOKENIZER_ID",
                f"{policy_path}@{model_manifest_revision(policy_path)}",
            ),
            current_policy_revision=os.getenv(
                "COEVO_CURRENT_POLICY_REVISION",
                model_manifest_revision(current_checkpoint),
            ),
            previous_policy_revision=os.getenv(
                "COEVO_PREVIOUS_POLICY_REVISION",
                model_manifest_revision(previous_checkpoint),
            ),
            buyer_revision=os.getenv(
                "COEVO_BUYER_REVISION",
                model_manifest_revision(buyer_checkpoint),
            ),
            teacher_gap_topk=int(os.getenv("COEVO_TEACHER_GAP_TOPK", "20")),
            teacher_gap_min_support_mass=float(
                os.getenv("COEVO_TEACHER_GAP_MIN_SUPPORT_MASS", "0.95")
            ),
            teacher_gap_eps=float(os.getenv("COEVO_TEACHER_GAP_EPS", "1e-8")),
            skill_gate_metric=os.getenv(
                "COEVO_SKILL_GATE_METRIC", "forward_kl"
            ),
            skill_gate_eps=float(os.getenv("COEVO_SKILL_GATE_EPS", "1e-8")),
            skill_gate_low=float(os.getenv("COEVO_SKILL_GATE_LOW", "0.0")),
            skill_gate_high=float(os.getenv("COEVO_SKILL_GATE_HIGH", "0.05")),
            skill_sharpen_t_min=float(
                os.getenv("COEVO_SKILL_SHARPEN_T_MIN", "0.7")
            ),
        )
