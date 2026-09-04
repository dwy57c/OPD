#!/usr/bin/env python3
import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from tau2.data_model.message import AssistantMessage

from coevo.audit import (
    BehaviorAuditor,
    ConditionalLeakageProbe,
    NLLeakageJudge,
    OpenAIGroundingJudge,
    build_matched_shuffled_examples,
    aggregate_session_signals,
)
from coevo.config import InfraConfig
from coevo.environment import Tau2Environment
from coevo.environment.tau2 import dump_messages, load_messages
from coevo.hints import HINT_LEVELS, HintLevel
from coevo.hinter_prompt import (
    narrow_privileged_context,
    student_profile_from_decision,
)
from coevo.hinter_training import (
    TeacherForcedUsefulnessScorer,
    build_hinter_grpo_dataset,
    calibrate_copying_weight,
)
from coevo.hinter_training.reward_dataset import REFERENCE_POOL_SOURCES
from coevo.intervention import DecisionState, TeacherActionGenerator
from coevo.models.hinted_teacher import (
    oracle_assistant_actions_from_task,
    oracle_steps_from_task,
)
from coevo.rollout import student_view


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def summarize_level_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("hint_error")]
    attempted = len(rows)
    count = len(valid) or 1
    scored = [row["analytical_signals"] for row in valid if row.get("analytical_signals")]
    result = {
        "rows": attempted,
        "valid_rows": len(valid),
        "hint_error_rows": attempted - len(valid),
        "contract_violation_rate": (
            (attempted - len(valid)) / attempted if attempted else 0.0
        ),
        "analytical_scored_rows": len(scored),
        "analytical_unscored_valid_rows": len(valid) - len(scored),
        "mean_hint_words": sum(row["hint_words"] for row in valid) / count,
        "clarification_rate": sum(
            row["behavior"]["clarification_rate"] for row in valid
        )
        / count,
        "lookup_rate": sum(row["behavior"]["lookup_rate"] for row in valid) / count,
        "ungrounded_assertion_rate": sum(
            row["behavior"]["ungrounded_assertion_rate"] for row in valid
        )
        / count,
    }
    if scored:
        result.update(
            {
                "mean_lift": sum(row["mean_lift"] for row in scored) / len(scored),
                "mean_copy": sum(row["mean_copy"] for row in scored) / len(scored),
                "mean_dose_kl": sum(row["mean_dose_kl"] for row in scored)
                / len(scored),
                "copy_fraction": sum(row["copy_fraction"] for row in scored)
                / len(scored),
                "transferable_fraction": sum(
                    row["transferable_fraction"] for row in scored
                )
                / len(scored),
            }
        )
    result["session_analytical"] = aggregate_session_signals(valid)
    return result


def _states_from_trajectories(path: Path) -> list[dict[str, Any]]:
    states = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        reward_info = record.get("reward_info")
        reward = (
            reward_info.get("reward")
            if isinstance(reward_info, dict)
            else record.get("student_trajectory_score")
        )
        verified = isinstance(reward, (int, float)) and float(reward) >= 1.0
        for decision in record.get("student_decisions", []):
            states.append(
                {
                    "domain": record.get("domain", "airline"),
                    "task_split": record.get("task_split", "train"),
                    "task_id": str(record["task_id"]),
                    "seed": int(record.get("seed", 42)),
                    "session_id": f"{record['task_id']}:{record.get('seed', 42)}",
                    "history_before": decision["history_before"],
                    "student_action": decision["student_action"],
                    "student_trajectory_score": reward,
                    "student_trajectory_verified": verified,
                }
            )
    return states


def _collect_states(
    config: InfraConfig,
    task_ids: list[str],
    *,
    evaluate_student: bool = False,
) -> list[dict[str, Any]]:
    states = []
    for task_id in task_ids:
        task_config = replace(
            config, task_id=task_id, hint_level=HintLevel.L0_NONE
        )
        environment = Tau2Environment(task_config)
        orchestrator = environment.orchestrator(
            environment.initial_history(), "student", seed=config.seed
        )
        simulation = orchestrator.run()
        reward = None
        if evaluate_student:
            reward = float(environment.evaluate(simulation).reward)
        history = simulation.get_messages()
        initial_size = len(environment.initial_history())
        for index in range(initial_size, len(history)):
            if not isinstance(history[index], AssistantMessage):
                continue
            decision = DecisionState.from_history(history, index)
            states.append(
                {
                    "domain": config.domain,
                    "task_split": config.task_split,
                    "task_id": task_id,
                    "seed": config.seed,
                    "session_id": f"{task_id}:{config.seed}",
                    "history_before": dump_messages(list(decision.history_before)),
                    "student_action": decision.student_action.model_dump(mode="json"),
                    "student_trajectory_score": reward,
                    "student_trajectory_verified": (
                        reward is not None and reward >= 1.0
                    ),
                }
            )
    return states


def _decision(state: dict[str, Any]) -> DecisionState:
    history = load_messages(state["history_before"])
    history.append(AssistantMessage.model_validate(state["student_action"]))
    return DecisionState.from_history(history, len(history) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 static audit of the L0-L3 hint ladder")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-profile-manifest", type=Path, required=True)
    parser.add_argument("--from-trajectories", type=Path)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--levels", nargs="+", choices=[x.value for x in HINT_LEVELS])
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument(
        "--use-grounding-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--run-leakage-probe", action="store_true")
    parser.add_argument(
        "--score-transfer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the three-view teacher-forced E1 copy/transfer decomposition.",
    )
    parser.add_argument(
        "--standard-source-level",
        choices=REFERENCE_POOL_SOURCES,
        default="oracle",
        help="Reference-pool source; retained under the old flag name for compatibility.",
    )
    args = parser.parse_args()
    if bool(args.from_trajectories) == bool(args.task_ids):
        parser.error("provide exactly one of --from-trajectories or --task-ids")

    config = InfraConfig.from_env()
    profile_manifest = json.loads(
        args.student_profile_manifest.read_text(encoding="utf-8")
    )
    profile_decisions = profile_manifest.get("decisions") or {}
    states = (
        _states_from_trajectories(args.from_trajectories)
        if args.from_trajectories
        else _collect_states(
            config,
            [str(value) for value in args.task_ids],
            evaluate_student=(args.standard_source_level != "oracle"),
        )
    )
    if args.max_states:
        states = states[: args.max_states]
    missing_profiles = sorted(
        {
            str(state["task_id"])
            for state in states
            if str(state["task_id"]) not in profile_decisions
        }
    )
    if missing_profiles:
        raise ValueError(
            "student-profile manifest is missing task decisions: "
            f"{missing_profiles}"
        )
    levels = [HintLevel.parse(value) for value in (args.levels or [x.value for x in HINT_LEVELS])]
    grounding = None
    if args.use_grounding_judge:
        grounding = OpenAIGroundingJudge(
            config.nl_judge or config.policy, retries=config.nl_judge_retries
        )
    auditor = BehaviorAuditor(grounding)
    transfer_scorer = (
        TeacherForcedUsefulnessScorer(config) if args.score_transfer else None
    )

    rows = []
    probe_sources = []
    teacher_generators = {}
    task_public_states = {}
    for state in states:
        task_public_states.setdefault(state["session_id"], state["history_before"])
    for state_index, state in enumerate(states):
        for level in levels:
            level_config = replace(
                config,
                domain=state["domain"],
                task_split=state["task_split"],
                task_id=state["task_id"],
                hint_level=level,
            )
            environment = Tau2Environment(level_config)
            decision = _decision(state)
            student = environment.policies.student(environment.fresh_environment())
            tools = [
                tool.openai_schema for tool in (getattr(student, "tools", None) or [])
            ]
            generator_key = (state["session_id"], level.value)
            generator = teacher_generators.setdefault(
                generator_key, TeacherActionGenerator(environment)
            )
            result = generator.generate(decision, state["seed"])
            audit_messages = [*decision.history_before, result.action]
            behavior = auditor.analyze(audit_messages)
            hint_note = ""
            if result.hint:
                hint_note = str((result.hint.get("hint") or {}).get("plan") or "")
            hint_error = (
                result.hint.get("error")
                if isinstance(result.hint, dict)
                else None
            )
            public_privilege = {
                "domain_policy": student.domain_policy,
                "authoritative_oracle_steps": oracle_steps_from_task(environment.task),
            }
            fact_audit_context = {
                "domain": state["domain"],
                "available_tools": tools,
                "authoritative_oracle_steps": public_privilege[
                    "authoritative_oracle_steps"
                ],
            }
            student_messages = student_view(
                student.system_prompt,
                [*decision.history_before, result.action],
            )
            row = {
                "state_id": f"{state['task_id']}:{state_index}",
                "state_order": state_index,
                "domain": state["domain"],
                "task_split": state["task_split"],
                "task_id": state["task_id"],
                "state_hash": decision.state_hash,
                "session_id": state["session_id"],
                "message_index": decision.message_index,
                "hint_level": level.value,
                "hint": result.hint,
                "hint_error": hint_error,
                "hint_words": len(hint_note.split()),
                "teacher_action": result.action.model_dump(mode="json"),
                "student_action": state["student_action"],
                "student_trajectory_score": state.get("student_trajectory_score"),
                "student_trajectory_verified": bool(
                    state.get("student_trajectory_verified")
                ),
                "oracle_reference_actions": [
                    action.model_dump(mode="json")
                    for action in oracle_assistant_actions_from_task(environment.task)
                ],
                "public_state": state["history_before"],
                "task_public_state": task_public_states[state["session_id"]],
                "student_profile": student_profile_from_decision(
                    profile_decisions[str(state["task_id"])]
                ),
                "privileged_context": narrow_privileged_context(public_privilege),
                "fact_audit_context": fact_audit_context,
                "student_visible_messages": student_messages,
                "tools": tools,
                "behavior": behavior.to_dict(),
                "analytical_signals": None,
            }
            rows.append(row)
            if not hint_error:
                probe_sources.append(
                    {
                        "state_id": row["state_id"],
                        "group_id": state["task_id"],
                        "task_id": state["task_id"],
                        "hint_level": level.value,
                        "state": state["history_before"],
                        "action": row["teacher_action"],
                        "hidden": oracle_steps_from_task(environment.task),
                    }
                )

    reference_rows = []
    if transfer_scorer is not None:
        reference_rows = build_hinter_grpo_dataset(
            rows,
            standard_source_level=args.standard_source_level,
            skip_missing_references=True,
        )
        for reference_row in reference_rows:
            serialized = reference_row.to_dict()
            session_rows = [
                row for row in rows if row["session_id"] == reference_row.session_id
            ]
            for level in levels:
                candidates = [
                    row
                    for row in session_rows
                    if row["hint_level"] == level.value and not row.get("hint_error")
                ]
                if not candidates:
                    continue
                hint = candidates[0].get("hint") or {}
                hint_note = str((hint.get("hint") or {}).get("plan") or "")
                if not hint_note:
                    continue
                candidates[0]["analytical_signals"] = (
                    transfer_scorer.score_reference_pool(
                        reference_pool=serialized["reference_pool"],
                        hint=hint_note,
                        tool_schemas=serialized["tools"],
                    ).to_dict()
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "audit_rows.jsonl", rows)
    _write_jsonl(args.output_dir / "probe_source_rows.jsonl", probe_sources)
    session_count = len({str(row["session_id"]) for row in rows})
    eligible_reference_sessions = len(reference_rows) if transfer_scorer else 0
    summary: dict[str, Any] = {
        "states": len(states),
        "levels": {},
        "analytical_scoring_panel": {
            "reference_pool_source": args.standard_source_level,
            "eligible_reference_sessions": eligible_reference_sessions,
            "skipped_sessions": session_count - eligible_reference_sessions,
            "skip_rate": (
                (session_count - eligible_reference_sessions) / session_count
                if session_count
                else 0.0
            ),
            "skip_reasons": {
                "missing_reference_trajectory": (
                    session_count - eligible_reference_sessions
                )
            },
        },
        "student_profile_manifest": str(args.student_profile_manifest),
    }
    for level in levels:
        selected = [row for row in rows if row["hint_level"] == level.value]
        summary["levels"][level.value] = summarize_level_rows(selected)
    l3 = summary["levels"].get(HintLevel.L3_ORACLE.value, {})
    l3_session = l3.get("session_analytical") or {}
    if (l3_session.get("mean_copy") or 0) > 0:
        summary["recommended_copying_weight_from_l3"] = calibrate_copying_weight(
            l3_mean_lift=l3_session["mean_lift"],
            l3_mean_copy=l3_session["mean_copy"],
        )

    if args.run_leakage_probe:
        judge = NLLeakageJudge(
            config.nl_judge or config.policy, retries=config.nl_judge_retries
        )
        probe_reports = {}
        for level in levels:
            selected = [row for row in probe_sources if row["hint_level"] == level.value]
            if len({row["task_id"] if "task_id" in row else row["state_id"].split(":")[0] for row in selected}) < 2:
                probe_reports[level.value] = {"skipped": "requires states from at least two tasks"}
                continue
            examples = build_matched_shuffled_examples(selected, seed=config.seed)
            probe = ConditionalLeakageProbe(
                lambda example: judge.score(example, include_action=True),
                lambda example: judge.score(example, include_action=False),
            )
            probe_reports[level.value] = probe.evaluate(examples).to_dict()
        summary["leakage_probe"] = probe_reports

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
