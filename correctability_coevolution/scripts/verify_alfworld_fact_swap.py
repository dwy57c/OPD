"""Replay recorded branches without policy calls and recheck grounding counts."""

import argparse
import json
from pathlib import Path

from experiment_alfworld_fact_swap import (
    ARMS, build_hint, new_env, pickup_evidence, public_state, sha, update_evidence, write_json,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result_dir', type=Path)
    args = p.parse_args()
    cases = json.loads((args.result_dir / 'cases.json').read_text())
    by_case = {c['case_id']: c for c in cases}
    branches = [json.loads(path.read_text()) for path in sorted((args.result_dir / 'branches').glob('*.json'))]
    assert len(branches) == len(cases) * len(ARMS)
    actual_steps, invalid, unsupported = 0, 0, 0
    for index, branch in enumerate(branches):
        case = by_case[branch['case_id']]
        env, state = new_env(case['game_file'], case['seed'])
        assert sha(public_state(state)) == case['initial_public_sha256']
        known, current = {}, None
        assert branch['hint'] == build_hint(branch['arm'], case['source'], case['alternative'])
        if branch['is_public_fact_control']:
            for step in case['source_reveal_history']:
                state, _, _ = env.step(step['action'])
                assert state.feedback == step['observation']
                current = update_evidence(known, step['action'], state.feedback, current)
        assert sha(public_state(state)) == branch['starting_public_sha256']
        count = 0
        for step in branch['steps']:
            assert public_state(state) == step['before']
            evidence = pickup_evidence(step['action'], known, state.admissible_commands)
            assert evidence == step['pickup_evidence']
            invalid += int(step['action'] not in state.admissible_commands)
            if evidence and evidence['object'].startswith('mug '):
                count += int(evidence['unverified'])
            state, reward, done = env.step(step['action'])
            assert state.feedback == step['observation']
            assert bool(done) == step['terminal']
            assert bool(state.won) == step['won']
            assert float(reward) == step['reward']
            current = update_evidence(known, step['action'], state.feedback, current)
            actual_steps += 1
        assert count == branch['metrics']['unverified_mug_pickup_attempts']
        unsupported += count
        env.close()
        if (index + 1) % 7 == 0:
            print(f'replayed {index+1}/{len(branches)} branches', flush=True)
    report = {'verified_branches':len(branches), 'executed_steps':actual_steps,
                      'invalid_actions':invalid, 'unverified_mug_pickup_attempts':unsupported,
                      'all_recorded_transitions_reproduced':True}
    write_json(args.result_dir / 'verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
