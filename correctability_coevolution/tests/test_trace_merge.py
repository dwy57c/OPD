import json
from pathlib import Path

import pytest

from coevo.training.trace_merge import merge_buyer_trace_shards


def _write_shard(path: Path, rank: int, rewards: list[float]) -> None:
    record = {
        "rank": rank,
        "world_size": 2,
        "expected_global_group_size": 4,
        "group_id": "prompt_0",
        "candidate_indexes": [0, 1],
        "raw_rewards": rewards,
        "post_skip_rewards": rewards,
        "rollouts": [{"candidate": f"{rank}:0"}, {"candidate": f"{rank}:1"}],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_merge_buyer_trace_shards_matches_global_group_normalization(tmp_path):
    rank0 = tmp_path / "trace.rank0.jsonl"
    rank1 = tmp_path / "trace.rank1.jsonl"
    _write_shard(rank0, 0, [0.0, 0.008])
    _write_shard(rank1, 1, [0.0, 0.0])

    records = merge_buyer_trace_shards([rank1, rank0])

    assert len(records) == 1
    record = records[0]
    assert record["raw_rewards"] == [0.0, 0.008, 0.0, 0.0]
    assert record["candidate_count"] == 4
    assert record["group_mean"] == pytest.approx(0.002)
    assert record["group_sample_std"] == pytest.approx(0.004)
    assert record["normalized_advantages"] == pytest.approx(
        [-0.487804878, 1.463414634, -0.487804878, -0.487804878]
    )
    assert [origin["rank"] for origin in record["candidate_origins"]] == [0, 0, 1, 1]
