from coevo.environment import Tau2Environment
from coevo.intervention import ActionBranchRunner


def build_action_branch_runner(environment: Tau2Environment) -> ActionBranchRunner:
    return ActionBranchRunner(
        environment,
        continuations=environment.config.continuations,
    )
