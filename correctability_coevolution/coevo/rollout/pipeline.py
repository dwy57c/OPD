from coevo.cutoff import TeacherCutoffSelector
from coevo.environment import Tau2Environment
from coevo.rewards import CorrectabilityEstimator
from coevo.rollout.cutoff_scorer import TurnCutoffScorer
from coevo.rollout.prefix_branch import PrefixBranchRunner


def build_cutoff_scorer(environment: Tau2Environment) -> TurnCutoffScorer:
    raw_environment = environment.fresh_environment()
    teacher = environment.policies.teacher(raw_environment, environment.task)
    selector = TeacherCutoffSelector(
        endpoint=environment.config.teacher,
        privileged_plan=teacher.make_agent_instructions_from_actions(),
        top_k=environment.config.cutoffs_per_turn,
        seed=environment.config.seed,
    )
    estimator = CorrectabilityEstimator(
        environment,
        continuations=environment.config.continuations,
        beta=environment.config.correctability_prior,
        continuation_runner=PrefixBranchRunner(environment),
    )
    return TurnCutoffScorer(selector, estimator)
