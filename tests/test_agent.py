import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import sqlite3
import tempfile
from traffic_agent import (Environment, posterior_congestion, train, run, recent_faults, precedent,
                           precedent_choice, CONFIG, FAULT_LIMIT)


class AgentTests(unittest.TestCase):
    def test_dwell_clearance_and_conservation(self):
        env = Environment(5)
        env.arrivals[:] = [2, 2]
        for step in range(env.steps):
            old_phase, old_age = env.phase, env.age
            _, _, info = env.step(1 - env.phase)
            if old_age < CONFIG['minimum_green_steps']:
                self.assertEqual(env.phase, old_phase)
            if info['switched']:
                self.assertEqual(info['departed'], 0)
                self.assertEqual(env.age, 0)
            self.assertTrue((env.queue >= 0).all())
        metrics = env.metrics()
        self.assertEqual(metrics['arrived_vehicles'], metrics['served_vehicles'] + metrics['remaining_vehicles'])

    def test_local_max_green_survives_api_failure(self):
        env = Environment(6)
        env.age = CONFIG['maximum_green_steps']
        env.api_failure[:] = True
        _, _, info = env.step(0)
        self.assertTrue(info['api_failure'])
        self.assertEqual(env.phase, 1)

    def test_paired_arrivals_are_controller_independent(self):
        a, b = Environment(42, 'sensor_fault'), Environment(42, 'sensor_fault')
        for _ in range(15):
            a.step(0)
            b.step(1)
        np.testing.assert_array_equal(a.arrivals, b.arrivals)
        np.testing.assert_array_equal(a.noise, b.noise)

    def test_bayesian_worked_example(self):
        # Predicted prior from .20 is .23. Positive posterior = .1955/.2725.
        self.assertAlmostEqual(posterior_congestion(.20, True), .1955 / .2725)

    def test_training_is_reproducible_and_updates_values(self):
        a, _, _ = train(.15, 22, episodes=3)
        b, _, _ = train(.15, 22, episodes=3)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.any(a != 0))

    def test_memory_safe_mode_counts_faults_and_matches_policy_without_faults(self):
        ok = {'observation': [5, 5], 'feedback': {'api_failure': False}}
        missing = {'observation': None, 'feedback': {'api_failure': False}}
        rejected = {'observation': [5, 5], 'feedback': {'api_failure': True}}
        self.assertEqual(recent_faults([ok, missing, rejected, ok]), 2)
        self.assertGreaterEqual(recent_faults([missing, rejected, missing]), FAULT_LIMIT)
        # With no faults the safe mode never triggers, so it equals the learned policy.
        q, _, _ = train(.15, 22, episodes=3)
        plain = run(q, 'balanced', 30001, 'q_learning')
        safe = run(q, 'balanced', 30001, 'q_learning_safe')
        self.assertEqual(plain['mean_queue_vehicles'], safe['mean_queue_vehicles'])

    def test_precedent_retrieval_backs_off_and_picks_better_action(self):
        db = sqlite3.connect(':memory:')
        db.execute('CREATE TABLE experience (ns, ew, phase, age, action, queue_change_20s, scenario, seed)')
        rows = [(4, 0, 0, a, 0, -3, 'x', 1) for a in range(5)] + [(4, 0, 0, 2, 1, 4, 'x', 1)] * 5
        db.executemany('INSERT INTO experience VALUES (?, ?, ?, ?, ?, ?, ?, ?)', rows)
        exact = precedent(db, (4, 0, 0, 2))      # only 1 case of action 0 at this exact age
        self.assertEqual(exact['match'], 'any green age')
        self.assertEqual(exact['cases'], [5, 5])
        self.assertEqual(precedent_choice(exact), 0)   # action 0 reduced the queue
        self.assertEqual(precedent(db, (0, 0, 0, 0))['match'], 'too few similar cases')
        self.assertIsNone(precedent_choice(precedent(db, (0, 0, 0, 0))))


if __name__ == '__main__':
    unittest.main()
