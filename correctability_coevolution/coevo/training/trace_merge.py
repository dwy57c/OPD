import json
import math
from pathlib import Path


def _sample_std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )


def merge_buyer_trace_shards(paths: list[Path]) -> list[dict]:
    """Reconstruct global GRPO groups from rank-local reward trace shards.

    Accelerate gathers reward tensors in rank order before TRL applies group
    normalization.  The reward callback itself runs before that gather, so its
    distributed trace must be persisted as one shard per rank and merged using
    the same ordering.
    """
    if not paths:
        raise ValueError("at least one Buyer trace shard is required")
    records_by_rank: dict[int, list[dict]] = {}
    for path in paths:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError(f"Buyer trace shard is empty: {path}")
        rank = int(records[0]["rank"])
        if any(int(record["rank"]) != rank for record in records):
            raise ValueError(f"mixed ranks in Buyer trace shard: {path}")
        if rank in records_by_rank:
            raise ValueError(f"duplicate Buyer trace rank: {rank}")
        records_by_rank[rank] = records

    ranks = sorted(records_by_rank)
    capture_counts = {len(records_by_rank[rank]) for rank in ranks}
    if len(capture_counts) != 1:
        raise ValueError("Buyer trace ranks have different capture counts")
    expected_world_size = int(records_by_rank[ranks[0]][0]["world_size"])
    if ranks != list(range(expected_world_size)):
        raise ValueError(
            f"Buyer trace ranks {ranks} do not cover world_size={expected_world_size}"
        )

    merged: list[dict] = []
    capture_count = capture_counts.pop()
    for capture_index in range(capture_count):
        shards = [records_by_rank[rank][capture_index] for rank in ranks]
        group_ids = sorted({str(shard["group_id"]) for shard in shards})
        for group_id in group_ids:
            group_shards = [
                shard for shard in shards if str(shard["group_id"]) == group_id
            ]
            raw_rewards: list[float] = []
            post_skip_rewards: list[float] = []
            rollouts: list[dict] = []
            origins: list[dict] = []
            for shard in group_shards:
                shard_rank = int(shard["rank"])
                indexes = list(shard["candidate_indexes"])
                if not (
                    len(indexes)
                    == len(shard["raw_rewards"])
                    == len(shard["post_skip_rewards"])
                    == len(shard["rollouts"])
                ):
                    raise ValueError("inconsistent Buyer trace shard lengths")
                raw_rewards.extend(float(value) for value in shard["raw_rewards"])
                post_skip_rewards.extend(
                    float(value) for value in shard["post_skip_rewards"]
                )
                rollouts.extend(shard["rollouts"])
                origins.extend(
                    {"rank": shard_rank, "local_candidate_index": int(index)}
                    for index in indexes
                )

            expected_group_sizes = {
                int(shard["expected_global_group_size"])
                for shard in group_shards
                if shard.get("expected_global_group_size") is not None
            }
            if len(expected_group_sizes) > 1:
                raise ValueError("conflicting expected Buyer group sizes")
            if expected_group_sizes and len(post_skip_rewards) != expected_group_sizes.pop():
                raise ValueError("incomplete distributed Buyer reward group")

            mean = sum(post_skip_rewards) / len(post_skip_rewards)
            sample_std = _sample_std(post_skip_rewards, mean)
            merged.append(
                {
                    "scope": "global-grpo-group",
                    "formula": "(post_skip_reward-group_mean)/(sample_std+1e-4)",
                    "capture_index": capture_index,
                    "group_id": group_id,
                    "world_size": expected_world_size,
                    "candidate_count": len(post_skip_rewards),
                    "candidate_origins": origins,
                    "raw_rewards": raw_rewards,
                    "post_skip_rewards": post_skip_rewards,
                    "group_mean": mean,
                    "group_sample_std": sample_std,
                    "normalized_advantages": [
                        (value - mean) / (sample_std + 1e-4)
                        for value in post_skip_rewards
                    ],
                    "rollouts": rollouts,
                }
            )
    return merged


def write_merged_buyer_trace(paths: list[Path], output_path: Path) -> list[dict]:
    records = merge_buyer_trace_shards(paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return records
