from copy import deepcopy

from openai import OpenAI
from tau2.data_model.message import AssistantMessage, Message

from coevo.environment import Tau2Environment
from coevo.models.hinted_teacher import HintedTeacherAgent
from coevo.rollout.views import student_view


class PrefixBranchRunner:
    """Continue an unfinished assistant message, then run the branch to terminal."""

    def __init__(self, environment: Tau2Environment):
        self.environment = environment

    def _complete_prefix(
        self, history: list[Message], policy: str, seed: int
    ) -> list[Message]:
        if not history or not isinstance(history[-1], AssistantMessage):
            raise ValueError("Prefix branch must end with a partial AssistantMessage")
        prefix = history[-1].content or ""
        raw_environment = self.environment.fresh_environment()
        if policy == "teacher":
            agent = self.environment.policies.teacher(
                raw_environment, self.environment.task_at(history)
            )
            if not isinstance(agent, HintedTeacherAgent):
                raise ValueError("Teacher prefix continuation requires closed-model hints")
            system_prompt, _ = agent.hinted_system_prompt_for_history(history)
        else:
            agent = self.environment.policies.student(raw_environment)
            system_prompt = agent.system_prompt
        endpoint = self.environment.config.policy

        messages = student_view(system_prompt, history)
        client = OpenAI(
            base_url=endpoint.base_url.rstrip("/") + "/v1", api_key="EMPTY"
        )
        response = client.chat.completions.create(
            model=endpoint.model,
            messages=messages,
            tool_choice="none",
            temperature=0.2,
            seed=seed,
            max_tokens=self.environment.config.branch_max_tokens,
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        continuation = response.choices[0].message.content
        if continuation is None:
            raise ValueError("Prefix continuation returned no text")
        completed = deepcopy(history)
        completed[-1] = completed[-1].model_copy(
            update={"content": prefix + continuation}
        )
        return completed

    def run(self, history: list[Message], policy: str, seed: int):
        completed = self._complete_prefix(history, policy, seed)
        return self.environment.continue_to_terminal(completed, policy, seed=seed)
