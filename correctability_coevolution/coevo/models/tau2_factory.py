from tau2.agent.llm_agent import LLMAgent, LLMGTAgent
from tau2.user.user_simulator import UserSimulator

from coevo.config import InfraConfig, ModelEndpoint


class Tau2PolicyFactory:
    """Construct τ² policies. It does not own an environment or rollout loop."""

    def __init__(self, config: InfraConfig):
        self.config = config

    @staticmethod
    def _agent(environment, endpoint: ModelEndpoint, task=None):
        kwargs = dict(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=endpoint.litellm_model,
            llm_args=endpoint.litellm_args,
        )
        if task is None:
            return LLMAgent(**kwargs)
        return LLMGTAgent(task=task, **kwargs)

    def student(self, environment):
        return self._agent(environment, self.config.student)

    def teacher(self, environment, task):
        return self._agent(environment, self.config.teacher, task=task)

    def buyer_reference(self, environment, task):
        user_tools = environment.get_user_tools() if environment.user_tools else None
        return UserSimulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=self.config.buyer_reference.litellm_model,
            llm_args=self.config.buyer_reference.litellm_args,
        )

