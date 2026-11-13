"""Reproducible teaching prototype. Synthetic queues, not a road safety system.

Run from the project folder: python src/traffic_agent.py experiment
Dependencies: numpy. Everything else comes with Python.
"""
from __future__ import annotations
import argparse
from collections import deque
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'config.json').read_text())
SCENARIOS = ('balanced', 'directional', 'rush_change', 'sensor_fault', 'capacity_loss')
BINS = np.array([3, 8, 15, 30])
# Seed of the saved demonstration traces used in the report. Other seeds give new traffic.
DEMO_SEED = 40001
# Short-term memory: the last 12 events (60 s). Three or more faults in that window
# (missing readings or rejected commands) switch the agent to the queue-rule safe mode,
# because the learned policy was weaker than the queue rule under sensor faults.
MEMORY_EVENTS = 12
FAULT_LIMIT = 3
# Extra training seeds for the robustness check; each run's 2,200 episode seeds stay
# below the validation (20001+) and test (30001+) seeds.
ROBUSTNESS_SEEDS = (5023, 8023, 11023, 14023, 17023)
# Long-term episodic memory: an indexed SQLite experience base built from validation
# episodes only (never test seeds). Each row is one situation, the action taken and the
# simulated change in total queue over the next 4 steps (20 s).
EXPERIENCE_DB = ROOT / 'results' / 'experience.sqlite'
OUTCOME_STEPS = 4
EXPERIENCE_EPSILON = 0.2   # some exploration so both actions have precedent
MIN_PRECEDENT = 5          # fewer matching cases than this: no precedent claim


def demand(scenario, steps):
    """Arrival rates are declared assumptions, in vehicles per five seconds."""
    rates = np.tile([1.8, 1.8], (steps, 1))
    if scenario == 'directional':
        rates[:] = [2.5, 0.8]
    elif scenario in ('rush_change', 'sensor_fault', 'capacity_loss'):
        rates[:steps // 2] = [2.5, 0.8]
        rates[steps // 2:] = [0.8, 2.5]
    return rates


class Environment:
    def __init__(self, seed, scenario='balanced', config=None):
        self.cfg = CONFIG if config is None else config
        self.scenario = scenario
        self.steps = self.cfg['episode_steps']
        # Separate random streams keep arrivals and sensor noise identical for
        # every controller on a paired seed, independent of its actions.
        self.arrivals = np.random.default_rng(seed).poisson(demand(scenario, self.steps))
        rng = np.random.default_rng(seed + 1000000)
        sigma = 4.0 if scenario == 'sensor_fault' else 1.0
        self.noise = rng.normal(0, sigma, (self.steps + 1, 2))
        self.missing = rng.random(self.steps + 1) < (0.20 if scenario == 'sensor_fault' else 0.0)
        self.api_failure = rng.random(self.steps) < (0.10 if scenario == 'sensor_fault' else 0.0)
        self.queue = np.zeros(2, dtype=int)
        self.phase, self.age, self.t = 0, 0, 0
        self.cumulative_queue, self.served, self.switches = 0, 0, 0
        self.max_queue, self.failures = 0, 0
        self.initial_vehicles = 0

    def observe(self):
        if self.missing[self.t]:
            return None
        return np.maximum(0, np.rint(self.queue + self.noise[self.t])).astype(int)

    def state(self, observation):
        if observation is None:
            return None
        bins = np.digitize(observation, BINS)
        return int(bins[0]), int(bins[1]), self.phase, min(self.age, 6)

    def step(self, requested):
        """Plant enforces dwell limits; switching consumes one all-red step.

        Clearance combines amber/all-red into a single teaching abstraction.
        API failure rejects a remote request, but the local max-green interlock
        still runs. No simultaneous opposing green is representable.
        """
        if requested not in (0, 1):
            raise ValueError('Action must be 0 (NS) or 1 (EW)')
        if self.t >= self.steps:
            raise RuntimeError('Episode ended')
        action = requested
        failure = bool(self.api_failure[self.t])
        if failure:
            action = self.phase
            self.failures += 1
        if self.age < self.cfg['minimum_green_steps']:
            action = self.phase
        if self.age >= self.cfg['maximum_green_steps']:
            action = 1 - self.phase
        self.queue += self.arrivals[self.t]
        switched = action != self.phase
        if switched:
            self.phase, self.age = action, 0
            self.switches += 1
            departed = 0
        else:
            service = self.cfg['service_per_step']
            if self.scenario == 'capacity_loss' and self.t >= self.steps // 2:
                service = 3
            departed = min(int(self.queue[self.phase]), service)
            self.queue[self.phase] -= departed
            self.age += 1
        self.served += departed
        total = int(self.queue.sum())
        self.cumulative_queue += total
        self.max_queue = max(self.max_queue, total)
        reward = -float(total) - self.cfg['switch_penalty'] * switched
        self.t += 1
        return reward, self.t == self.steps, {'switched': bool(switched), 'api_failure': failure, 'departed': departed}

    def metrics(self):
        arrived = int(self.arrivals[:self.t].sum())
        assert self.initial_vehicles + arrived == self.served + int(self.queue.sum())
        return {
            'mean_queue_vehicles': self.cumulative_queue / self.t,
            # This is total queue exposure per arrival over the observed horizon.
            # It is NOT mean completed-trip delay; remaining queues are censored.
            'queue_exposure_seconds_per_arrival': self.cumulative_queue * self.cfg['step_seconds'] / max(1, arrived),
            'served_vehicles': self.served,
            'arrived_vehicles': arrived,
            'remaining_vehicles': int(self.queue.sum()),
            'max_total_queue': self.max_queue,
            'switches': self.switches,
            'api_failures': self.failures
        }


def baseline(env, observation, name='fixed'):
    if name == 'fixed' or observation is None:
        return env.phase if env.age < 4 else 1 - env.phase
    # Hysteresis avoids paying a clearance cost for tiny queue differences.
    return 1 - env.phase if observation[1 - env.phase] > observation[env.phase] + 3 else env.phase


def is_fault(event):
    """A fault is a missing reading or a command the signal controller rejected."""
    return event['observation'] is None or event['feedback']['api_failure']


def recent_faults(memory):
    return sum(1 for event in memory if is_fault(event))


def choose(q, state, rng=None, epsilon=0):
    if rng is not None and rng.random() < epsilon:
        return int(rng.integers(2))
    values = q[state]
    # Deterministic tie-break holds the current phase, avoiding seed-dependent
    # random evaluation and unnecessary clearance when values are equal.
    return state[2] if values[0] == values[1] else int(np.argmax(values))


def train(alpha, seed, episodes=None, config=None):
    q = np.zeros((5, 5, 2, 7, 2), dtype=float)
    visits = np.zeros(q.shape[:-1], dtype=np.int64)
    rng = np.random.default_rng(seed)
    records = []
    count = CONFIG['training_episodes'] if episodes is None else episodes
    for ep in range(count):
        env = Environment(seed + ep, SCENARIOS[ep % 3], config)
        epsilon = max(CONFIG['epsilon_end'], CONFIG['epsilon_start'] -
                      (CONFIG['epsilon_start'] - CONFIG['epsilon_end']) * ep / CONFIG['epsilon_decay_episodes'])
        obs = env.observe()
        total_reward = 0
        for _ in range(env.steps):
            state = env.state(obs)
            action = choose(q, state, rng, epsilon)
            reward, done, _ = env.step(action)
            next_obs = env.observe()
            next_state = env.state(next_obs)
            target = reward if done else reward + CONFIG['discount'] * np.max(q[next_state])
            q[state + (action,)] += alpha * (target - q[state + (action,)])
            visits[state] += 1
            obs = next_obs
            total_reward += reward
        if ep % 50 == 0 or ep == count - 1:
            records.append({'episode': ep + 1, 'alpha': alpha, 'epsilon': epsilon,
                            'reward': total_reward, **env.metrics()})
    return q, visits, records


def run(q, scenario, seed, controller):
    """controller: fixed, greedy (queue rule), q_learning, or q_learning_safe
    (Q-learning plus the short-term-memory safe mode, without a human gate)."""
    env = Environment(seed, scenario)
    memory = deque(maxlen=MEMORY_EVENTS)
    for _ in range(env.steps):
        obs = env.observe()
        if controller == 'q_learning' and obs is not None:
            action = choose(q, env.state(obs))
        elif controller == 'q_learning_safe' and obs is not None:
            action = (baseline(env, obs, 'greedy') if recent_faults(memory) >= FAULT_LIMIT
                      else choose(q, env.state(obs)))
        else:
            action = baseline(env, obs, controller)
        _, _, feedback = env.step(action)
        memory.append({'observation': obs, 'feedback': feedback})
    return {'scenario': scenario, 'seed': seed, 'controller': controller, **env.metrics()}


def build_experience(q, path=EXPERIENCE_DB):
    """Fill the experience base from validation episodes and index it by situation."""
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE experience (ns INTEGER, ew INTEGER, phase INTEGER, age INTEGER, '
               'action INTEGER, queue_change_20s INTEGER, scenario TEXT, seed INTEGER)')
    rows = []
    for scenario in SCENARIOS:
        for seed in CONFIG['validation_seeds']:
            env = Environment(seed, scenario)
            rng = np.random.default_rng(seed)
            steps = []
            for _ in range(env.steps):
                before = int(env.queue.sum())
                obs = env.observe()
                state = env.state(obs)
                action = choose(q, state, rng, EXPERIENCE_EPSILON) if state is not None else baseline(env, obs)
                env.step(action)
                steps.append((state, action, before, int(env.queue.sum())))
            for t, (state, action, before, _) in enumerate(steps[:-OUTCOME_STEPS + 1 or None]):
                if state is not None and t + OUTCOME_STEPS - 1 < len(steps):
                    after = steps[t + OUTCOME_STEPS - 1][3]
                    rows.append((*state, action, after - before, scenario, seed))
    db.executemany('INSERT INTO experience VALUES (?, ?, ?, ?, ?, ?, ?, ?)', rows)
    db.execute('CREATE INDEX situation ON experience (ns, ew, phase, age)')
    db.commit()
    db.close()
    return len(rows)


def precedent(db, state):
    """Retrieve past events in a similar discretised situation (indexed key lookup, not a
    vector search). Back-off: exact situation, then any green age, then queues only, until
    both actions have enough cases. Returns per action: cases and mean 20 s queue change."""
    out = {'match': None, 'cases': [0, 0], 'mean_queue_change_20s': [None, None]}
    if db is None or state is None:
        return out
    levels = (('exact', 'ns=? AND ew=? AND phase=? AND age=?', state),
              ('any green age', 'ns=? AND ew=? AND phase=?', state[:3]),
              ('queues only', 'ns=? AND ew=?', state[:2]))
    for name, where, key in levels:
        cases, means = [0, 0], [None, None]
        for action, count, mean in db.execute(
                f'SELECT action, COUNT(*), AVG(queue_change_20s) FROM experience WHERE {where} GROUP BY action', key):
            cases[action], means[action] = count, round(mean, 2)
        out = {'match': name, 'cases': cases, 'mean_queue_change_20s': means}
        if min(cases) >= MIN_PRECEDENT:
            break
    else:
        out['match'] = 'too few similar cases'
    return out


def precedent_choice(p):
    """Action whose past cases reduced the queue more, if both have enough cases."""
    if min(p['cases']) < MIN_PRECEDENT:
        return None
    m = p['mean_queue_change_20s']
    return 0 if m[0] <= m[1] else 1


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def posterior_congestion(prior, detector_positive):
    """Illustrative detector sensitivity=.85, false-positive rate=.10.

    Parameters are assumptions, not empirically calibrated probabilities.
    Prediction step permits a latent regime change between observations.
    """
    predicted = 0.90 * prior + 0.05
    likelihood_c, likelihood_n = ((0.85, 0.10) if detector_positive else (0.15, 0.90))
    return predicted * likelihood_c / (predicted * likelihood_c + (1 - predicted) * likelihood_n)


def demo(approve=False, on_event=None, output_dir=None, readings=36, seed=DEMO_SEED):
    """Run the saved policy. An optional callback displays each completed step.

    The default seed reproduces the report's demonstration; another seed gives
    different arrivals, sensor noise, missing readings and command failures.
    """
    if not 1 <= readings <= CONFIG['episode_steps']:
        raise ValueError(f"readings must be between 1 and {CONFIG['episode_steps']}")
    out = ROOT / 'results' if output_dir is None else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    q = np.load(ROOT / 'results' / 'q_table.npy')
    env = Environment(seed, 'sensor_fault')
    # Artificial emergency starting queue gives a reproducible human-gate example.
    env.queue[:] = [45, 8]
    env.initial_vehicles = 53
    memory = deque(maxlen=MEMORY_EVENTS)
    experience = sqlite3.connect(f'file:{EXPERIENCE_DB}?mode=ro', uri=True) if EXPERIENCE_DB.exists() else None
    prior = 0.20
    db = sqlite3.connect(out / 'agent_memory.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS events (run_id TEXT, step INTEGER, payload TEXT)')
    # Feedback pipeline: every review outcome is stored with its reason for later audit.
    db.execute('CREATE TABLE IF NOT EXISTS operator_decisions (run_id TEXT, step INTEGER, decision TEXT, '
               'reason TEXT, triggers TEXT)')
    run_id = str(time.time_ns())
    with (out / 'demo_trace.jsonl').open('w') as log:
        for _ in range(readings):
            obs = env.observe()
            if obs is not None:
                prior = posterior_congestion(prior, int(obs.sum()) >= 30)
            else:
                prior = 0.90 * prior + 0.05
            state = env.state(obs)
            proposed = choose(q, state) if state is not None else baseline(env, obs)
            # Short-term memory is read before acting: repeated recent faults mean the
            # readings are unreliable, so the learned action is not used at all.
            faults = recent_faults(memory)
            past = precedent(experience, state)
            past_choice = precedent_choice(past)
            safe_mode = state is not None and faults >= FAULT_LIMIT
            # Review triggers: heavy queue, high congestion belief, or the learned action
            # contradicts what worked in similar past situations (long-term memory).
            triggers = [] if state is None or safe_mode else [
                name for name, hit in (('queue >= 40', int(obs.sum()) >= 40),
                                       ('P(congestion) > 0.90', prior > 0.90),
                                       ('precedent disagrees', past_choice is not None and past_choice != proposed))
                if hit]
            needs_review = bool(triggers)
            if obs is None:
                actual_request, reason = proposed, 'sensor missing, fixed timing fallback'
            elif safe_mode:
                actual_request = baseline(env, obs, 'greedy')
                reason = f'{faults} faults in last {MEMORY_EVENTS} events, queue-rule safe mode'
            elif needs_review and not approve:
                actual_request, reason = baseline(env, obs, 'greedy'), 'review required, queue-rule fallback'
            elif needs_review:
                actual_request, reason = proposed, 'simulated supervisor approval'
            else:
                actual_request, reason = proposed, 'learned action within local interlocks'
            row = {'step': env.t, 'observation': None if obs is None else obs.tolist(),
                   'congestion_probability_assumed_model': round(prior, 6),
                   'q_values': None if state is None else q[state].tolist(),
                   'proposed_phase': proposed, 'requested_phase': actual_request,
                   'review_required': bool(needs_review), 'simulated_approval': bool(approve and needs_review),
                   'reason': reason, 'memory_items_before': len(memory),
                   'recent_faults_in_memory': faults, 'safe_mode': bool(safe_mode),
                   'precedent': past, 'precedent_choice': past_choice, 'review_triggers': triggers}
            reward, _, feedback = env.step(actual_request)
            row.update({'reward': reward, 'acknowledged_phase': env.phase, 'green_age': env.age,
                        'feedback': feedback, 'true_queues_for_simulation_audit': env.queue.tolist()})
            memory.append(row)
            # SQLite is an auditable knowledge base, not a vector database.
            db.execute('INSERT INTO events VALUES (?, ?, ?)', (run_id, row['step'], json.dumps(row)))
            if needs_review:
                db.execute('INSERT INTO operator_decisions VALUES (?, ?, ?, ?, ?)',
                           (run_id, row['step'], 'approved' if approve else 'withheld',
                            'simulated: --approve flag' if approve else 'simulated: no approval given',
                            ', '.join(triggers)))
            db.commit()
            log.write(json.dumps(row) + '\n')
            if on_event is not None:
                on_event(row)
    count = db.execute('SELECT count(*) FROM events WHERE run_id=?', (run_id,)).fetchone()[0]
    db.close()
    if experience is not None:
        experience.close()
    print(json.dumps({'demo_events_persisted': count, 'short_term_events_retained': len(memory),
                      'simulated_approval': approve, 'seed': seed}, indent=2))


CONTROLLERS = ('fixed', 'greedy', 'q_learning', 'q_learning_safe')


def penalty_ablation(alpha):
    """Retrain the selected rate with switch penalty 0 instead of 2 and test on the same
    held-out seeds. Only the reward changes; the test metric (mean queue) is unchanged."""
    test_seeds = range(CONFIG['test_seed_start'], CONFIG['test_seed_start'] + CONFIG['test_seed_count'])
    rows = []
    for penalty in (CONFIG['switch_penalty'], 0.0):
        q, _, _ = train(alpha, CONFIG['seed'], config={**CONFIG, 'switch_penalty': penalty})
        for scenario in SCENARIOS:
            results = [run(q, scenario, seed, 'q_learning') for seed in test_seeds]
            rows.append({'switch_penalty': penalty, 'scenario': scenario,
                         'mean_queue_vehicles': float(np.mean([r['mean_queue_vehicles'] for r in results])),
                         'mean_switches': float(np.mean([r['switches'] for r in results]))})
    write_csv(ROOT / 'results' / 'penalty_ablation.csv', rows)
    print('switch penalty ablation done', flush=True)
    return rows


def training_seed_check(alpha):
    """Retrain the selected rate on five training seeds and test each policy on the
    same held-out seeds, to measure how much results depend on one training run."""
    test_seeds = range(CONFIG['test_seed_start'], CONFIG['test_seed_start'] + CONFIG['test_seed_count'])
    rows = []
    for training_seed in ROBUSTNESS_SEEDS:
        q, _, _ = train(alpha, training_seed)
        for scenario in SCENARIOS:
            for controller in ('q_learning', 'q_learning_safe'):
                values = [run(q, scenario, seed, controller)['mean_queue_vehicles'] for seed in test_seeds]
                rows.append({'training_seed': training_seed, 'scenario': scenario, 'controller': controller,
                             'mean_queue_vehicles': float(np.mean(values))})
        print(f'training seed {training_seed} checked', flush=True)
    write_csv(ROOT / 'results' / 'training_seeds.csv', rows)
    summary = []
    for scenario in SCENARIOS:
        for controller in ('q_learning', 'q_learning_safe'):
            values = np.array([r['mean_queue_vehicles'] for r in rows
                               if r['scenario'] == scenario and r['controller'] == controller])
            summary.append({'scenario': scenario, 'controller': controller, 'training_seeds': len(values),
                            'mean': float(values.mean()), 'sd': float(values.std(ddof=1)),
                            'min': float(values.min()), 'max': float(values.max())})
    return summary


def experiment():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    validation, logs, candidates = [], [], []
    started = time.monotonic()
    for alpha in CONFIG['learning_rates']:
        q, visits, records = train(alpha, CONFIG['seed'])
        scores = [run(q, scenario, seed, 'q_learning')['mean_queue_vehicles']
                  for scenario in SCENARIOS[:3] for seed in CONFIG['validation_seeds']]
        score = float(np.mean(scores))
        validation.append({'learning_rate': alpha, 'validation_mean_queue_vehicles': score})
        candidates.append((score, alpha, q, visits))
        logs.extend(records)
        print(f'alpha={alpha}: validation queue={score:.3f}', flush=True)
    _, alpha, q, visits = min(candidates, key=lambda item: item[0])
    np.save(out / 'q_table.npy', q)
    experience_rows = build_experience(q)
    np.save(out / 'state_visits.npy', visits)
    rows = [run(q, scenario, seed, controller)
            for scenario in SCENARIOS
            for seed in range(CONFIG['test_seed_start'], CONFIG['test_seed_start'] + CONFIG['test_seed_count'])
            for controller in CONTROLLERS]
    write_csv(out / 'evaluation.csv', rows)
    write_csv(out / 'training_log.csv', logs)
    write_csv(out / 'validation.csv', validation)
    summary = []
    for scenario in SCENARIOS:
        fixed = [r['mean_queue_vehicles'] for r in rows if r['scenario'] == scenario and r['controller'] == 'fixed']
        for controller in CONTROLLERS:
            selected = [r for r in rows if r['scenario'] == scenario and r['controller'] == controller]
            values = np.array([r['mean_queue_vehicles'] for r in selected])
            diffs = values - np.array(fixed)
            # Approximate paired 95% interval, not a claim of normality or field validity.
            se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
            summary.append({'scenario': scenario, 'controller': controller,
                            'mean_queue_vehicles': float(values.mean()),
                            'std_queue_vehicles': float(values.std(ddof=1)),
                            'mean_queue_exposure_seconds_per_arrival': float(np.mean([r['queue_exposure_seconds_per_arrival'] for r in selected])),
                            'mean_served_vehicles': float(np.mean([r['served_vehicles'] for r in selected])),
                            'paired_difference_vs_fixed': float(diffs.mean()),
                            'paired_95_low': float(diffs.mean() - 1.96 * se),
                            'paired_95_high': float(diffs.mean() + 1.96 * se)})
    robustness = training_seed_check(alpha)
    ablation = penalty_ablation(alpha)
    result = {'selected_learning_rate': alpha, 'config': CONFIG, 'validation': validation,
              'training_seed_count': 1, 'training_seed_check': robustness,
              'experience_base_rows': experience_rows, 'switch_penalty_ablation': ablation, 'test_episodes_per_scenario_controller': CONFIG['test_seed_count'],
              'summary': summary, 'elapsed_seconds': round(time.monotonic() - started, 2),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'numpy_version': np.__version__,
              'scope': 'Synthetic queue simulation. Policy benchmarks exclude human approval gate. Demo adds that gate.'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'selected_learning_rate': alpha, 'elapsed_seconds': result['elapsed_seconds']}, indent=2))
    demo()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['experiment', 'demo'])
    parser.add_argument('--approve', action='store_true', help='Simulate approval in demo only; never a real supervisor approval')
    parser.add_argument('--seed', default=str(DEMO_SEED),
                        help=f'Demo only: traffic seed (default {DEMO_SEED}, the report run), or "time" for a new seed from the clock')
    args = parser.parse_args()
    if args.seed == 'time':
        demo_seed = int(time.time()) % 100000
    elif args.seed.isdigit():
        demo_seed = int(args.seed)
    else:
        parser.error('--seed must be a non-negative whole number or "time"')
    if args.command == 'experiment':
        if args.seed != str(DEMO_SEED):
            parser.error('--seed applies to demo only; experiment uses the seeds in config.json')
        experiment()
    else:
        demo(args.approve, seed=demo_seed)
