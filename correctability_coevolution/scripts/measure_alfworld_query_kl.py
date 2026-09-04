"""Paired full-vocabulary KL with a fixed hint and reference action prefix.

This is an offline diagnostic on saved ALFWorld expert decision states. It
does not generate hints, execute agent rollouts, or modify the GRPO reward.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch


SYSTEM = (
    "You are an ALFWorld household agent. Choose exactly one next command from "
    "the admissible command list. Return only that command, with no explanation."
)
LEVELS = ("L0_NONE", "L1_POLICY", "L2_PROCEDURAL", "L3_ORACLE")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def query_text(session, state):
    # Match the preceding E1 exactly; history is the last three exchanges.
    transcript = [f"{m['role']}: {m['content']}" for m in state["history"][-6:]]
    return (f"Goal: {session['goal']}\n" + "\n".join(transcript)
            + f"\nCurrent observation: {state['observation']}\n"
            + "Admissible commands:\n" + "\n".join(state["admissible_commands"]))


def divergence(q_logits, r_logits):
    """KL(Q||R), KL(R||Q), and JSD on the entire vocabulary, in nats."""
    q_log = torch.log_softmax(q_logits.float(), dim=-1)
    r_log = torch.log_softmax(r_logits.float(), dim=-1)
    if not torch.isfinite(q_log).all() or not torch.isfinite(r_log).all():
        raise ValueError("non-finite logits")
    mixture = torch.logaddexp(q_log, r_log) - math.log(2)
    forward = (q_log.exp() * (q_log - r_log)).sum(-1)
    reverse = (r_log.exp() * (r_log - q_log)).sum(-1)
    js = 0.5 * ((q_log.exp() * (q_log - mixture)).sum(-1)
                + (r_log.exp() * (r_log - mixture)).sum(-1))
    if min(forward.min().item(), reverse.min().item(), js.min().item()) < -1e-5:
        raise ValueError("negative KL beyond roundoff")
    return forward.clamp_min(0), reverse.clamp_min(0), js.clamp_min(0), q_log, r_log


def self_test():
    q = torch.tensor([[0.5, 0.5], [0.25, 0.75]]).log()
    r = torch.tensor([[0.25, 0.75], [0.25, 0.75]]).log()
    f, b, js, _, _ = divergence(q, r)
    expected = 0.5 * math.log(2) + 0.5 * math.log(2 / 3)
    assert abs(f[0].item() - expected) < 1e-6
    assert f[1].item() < 1e-6
    assert b[1].item() < 1e-6
    assert all(0 <= x <= math.log(2) for x in js.tolist())
    print("full-vocabulary KL identity and direction checks passed", flush=True)


def mean(values):
    return sum(values) / len(values) if values else None


METRICS = ("query_kl", "reverse_kl", "js_divergence", "first_token_query_kl")


def group_means(rows):
    return {key: mean([row[key] for row in rows]) for key in METRICS}


def summarize(rows, sessions):
    per_session = []
    for session in sessions:
        sid = session["session_id"]
        pickup = next((s["turn"] for s in session["states"]
                       if s["expert_action"].startswith("take ")), None)
        for level in LEVELS:
            selected = [r for r in rows if r["session_id"] == sid and r["level"] == level]
            if not selected:
                continue
            by_phase = {}
            for phase, subset in (
                ("initial", [r for r in selected if r["turn"] == 0]),
                ("before_pickup", [r for r in selected if pickup is not None and r["turn"] < pickup]),
                ("pickup_onwards", [r for r in selected if pickup is not None and r["turn"] >= pickup]),
            ):
                by_phase[phase] = {"turns": len(subset), **group_means(subset)}
            per_session.append({"session_id": sid, "level": level,
                                "turns": len(selected), **group_means(selected),
                                "phases": by_phase})
    levels = {}
    for level in LEVELS:
        selected = [r for r in per_session if r["level"] == level]
        phases = {}
        for phase in ("initial", "before_pickup", "pickup_onwards"):
            phases[phase] = {
                key: mean([r["phases"][phase][key] for r in selected
                           if r["phases"][phase][key] is not None])
                for key in METRICS
            }
        levels[level] = {"sessions": len(selected), **group_means(selected), "phases": phases}
    paired = []
    for session in sessions:
        selected = {r["level"]: r for r in per_session if r["session_id"] == session["session_id"]}
        if "L2_PROCEDURAL" in selected and "L3_ORACLE" in selected:
            l2, l3 = selected["L2_PROCEDURAL"], selected["L3_ORACLE"]
            paired.append({
                "session_id": session["session_id"],
                "l3_minus_l2_query_kl": l3["query_kl"] - l2["query_kl"],
                "l3_less_than_l2": l3["query_kl"] < l2["query_kl"],
                "initial_l3_minus_l2_query_kl": (
                    l3["phases"]["initial"]["query_kl"]
                    - l2["phases"]["initial"]["query_kl"]),
            })
    return {"levels": levels, "per_session": per_session, "paired_l3_vs_l2": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="/models/policy")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if not args.input_dir or not args.output_dir:
        parser.error("--input-dir and --output-dir are required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from coevo.models.hinted_teacher import format_teacher_system_prompt_with_hint

    sessions = read_rows(args.input_dir / "sessions.jsonl")
    hint_rows = read_rows(args.input_dir / "task_hints.jsonl")
    old_rows = read_rows(args.input_dir / "audit_rows.jsonl")
    hints = {(r["session_id"], r["level"]): r for r in hint_rows}
    old_targets = {(r["session_id"], r["turn"], r["hint_level"]):
                   r["analytical_signals"]["probability_trace"]["target_token_ids"]
                   for r in old_rows}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
    ).eval()
    print(f"loaded model, vocabulary={model.config.vocab_size}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def prefix(messages):
        ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                             add_generation_prompt=True, enable_thinking=False)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    @torch.inference_mode()
    def logits(prompt_ids, target_ids):
        # The last prompt token predicts the first action token. No target
        # terminators are scored, and every view has the same action prefix.
        ids = torch.tensor([prompt_ids + target_ids[:-1]], device="cuda:0")
        result = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       use_cache=False, logits_to_keep=len(target_ids))
        return result.logits[0].detach().clone()

    start = time.monotonic()
    rows = []
    rejected = []
    total = sum(len(s["states"]) for s in sessions) * len(LEVELS)
    cached_r = {}
    with (args.output_dir / "token_rows.jsonl").open("w", encoding="utf-8") as handle:
        for session in sessions:
            sid = session["session_id"]
            for level in LEVELS:
                record = hints.get((sid, level)) if level != "L0_NONE" else None
                if level != "L0_NONE" and (not record or record.get("error")):
                    rejected.append({"session_id": sid, "level": level, "reason": "missing/invalid hint"})
                    continue
                hint = record["hint"]["plan"] if record else ""
                system = format_teacher_system_prompt_with_hint(SYSTEM, {"plan": hint}) if hint else SYSTEM
                r_prefix = prefix([{"role": "system", "content": system}])
                for state in session["states"]:
                    target = state["expert_action"]
                    target_ids = tokenizer.encode(target, add_special_tokens=False)
                    if level != "L0_NONE":
                        if target_ids != old_targets[(sid, state["turn"], level)]:
                            raise ValueError("target IDs differ from original E1")
                    q_prefix = prefix([{"role": "system", "content": system},
                                       {"role": "user", "content": query_text(session, state)}])
                    q_logits = logits(q_prefix, target_ids)
                    cache_key = (digest(system), tuple(target_ids))
                    if cache_key not in cached_r:
                        cached_r[cache_key] = logits(r_prefix, target_ids).cpu()
                    r_logits = cached_r[cache_key].to("cuda:0")
                    f, b, js, q_log, r_log = divergence(q_logits, r_logits)
                    indices = torch.tensor(target_ids, device="cuda:0")[:, None]
                    q_actual = q_log.gather(-1, indices).squeeze(-1).tolist()
                    r_actual = r_log.gather(-1, indices).squeeze(-1).tolist()
                    row = {
                        "session_id": sid, "level": level, "turn": state["turn"],
                        "standard_action": target, "hint_sha256": digest(hint),
                        "query_sha256": digest(query_text(session, state)),
                        "target_ids": target_ids, "target_tokens": tokenizer.convert_ids_to_tokens(target_ids),
                        "query_prompt_tokens": len(q_prefix), "hint_only_prompt_tokens": len(r_prefix),
                        "query_kl": f.mean().item(), "reverse_kl": b.mean().item(),
                        "js_divergence": js.mean().item(), "first_token_query_kl": f[0].item(),
                        "token_query_kl": f.tolist(), "token_reverse_kl": b.tolist(),
                        "token_js": js.tolist(), "query_actual_logprobs": q_actual,
                        "hint_only_actual_logprobs": r_actual,
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    if len(rows) % 25 == 0:
                        print(f"scored {len(rows)}/{total}, elapsed={time.monotonic()-start:.1f}s", flush=True)
                print(f"finished {sid} {level}", flush=True)
    summary = {
        "metric": "KL(p_theta(.|s,h,a_<t) || p_theta(.|h,a_<t))",
        "vocabulary": model.config.vocab_size,
        "computation": "full vocabulary; BF16 model forward, FP32 log_softmax and KL; no top-k or clipping",
        "aggregation": "mean tokens within action, mean turns within session, equal session mean",
        "query_definition": "goal + last 6 history messages + current observation + admissible command list",
        "reference_source": "ALFWorld built-in rule-expert trace saved by preceding E1",
        "hint_model": "glm-5.3-flash (unchanged saved hints)", "policy": args.model,
        "limitations": ["3 different games, one session each; stored seed labels were not applied to the environment",
                        "teacher-forced expert states, no autonomous policy rollout",
                        "one oracle reference per session; successful-student reference pool not used"],
        "input_sha256": {name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
                         for name in ("sessions.jsonl", "task_hints.jsonl", "audit_rows.jsonl")},
        "rows": len(rows), "expected_rows": total, "rejected_session_levels": rejected,
        "elapsed_seconds": time.monotonic() - start,
        **summarize(rows, sessions),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"levels": summary["levels"], "paired": summary["paired_l3_vs_l2"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
