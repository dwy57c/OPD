from tau2.agent.llm_agent import LLMAgent
from tau2.user.user_simulator import UserSimulator

from coevo.config import InfraConfig, ModelEndpoint
from coevo.models.hinted_teacher import HintedTeacherAgent


class Tau2PolicyFactory:
    """Construct τ² policies. It does not own an environment or rollout loop."""

    def __init__(self, config: InfraConfig):
        self.config = config

    @staticmethod
    def _agent(environment, endpoint: ModelEndpoint):
        return LLMAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=endpoint.litellm_model,
            llm_args=endpoint.litellm_args,
        )

    def student(self, environment):
        return self._agent(environment, self.config.policy)

    def teacher(self, environment, task, hint_result=None):
        if self.config.teacher_hint_mode == "closed_model":
            return HintedTeacherAgent(
                tools=environment.get_tools(),
                domain_policy=environment.get_policy(),
                task=task,
                llm=self.config.policy.litellm_model,
                llm_args=self.config.policy.litellm_args,
                hinter_endpoint=self.config.teacher_hinter,
                initial_hint=hint_result,
            )
        return self._agent(environment, self.config.policy)

    def buyer_reference(self, environment, task):
        user_tools = environment.get_user_tools() if environment.user_tools else None
        return UserSimulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=self.config.buyer_reference.litellm_model,
            llm_args=self.config.buyer_reference.litellm_args,
        )
