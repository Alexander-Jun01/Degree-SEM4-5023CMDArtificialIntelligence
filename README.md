# Traffic signal Q-learning agent

Tabular Q-learning agent for one simulated traffic intersection, written for the 5023CMD Artificial Intelligence CW2 group project (smart city traffic management option). The repository contains the agent, its tests, the saved model and results, and the scripts that draw the report figures.

The agent requests north-south or east-west green every five seconds. A supervisor layer adds a Bayesian congestion check, a short-term-memory safe mode, precedent retrieval from past experience, simulated human approval and a SQLite audit log. All traffic is synthetic.

## Install

Python 3.10 or newer. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements.txt` installs everything the project uses: NumPy for the agent, Matplotlib and SciPy for the report figures, and TensorBoard for the optional result viewer. To reproduce the saved results with the exact versions used, install `requirements-lock.txt` instead.

## Lab-style simulation (start here)

The scripts follow the SmartFactory lab's layout: settings at the top, commented sections, printed readings and a CSV in `data`. The topic remains traffic lights and the existing Q-learning agent makes the decisions.

From the TrafficAgent folder, run:

```bash
python sensor_simulation.py
python data_analysis.py
```

The simulator prints 36 readings with a half-second pause between them. Each reading represents five simulated seconds. Change `NUMBER_OF_READINGS` (1 to 360), `TIME_BETWEEN_READINGS`, or `SIMULATED_APPROVAL` at the top of `sensor_simulation.py`. Approval is simulated, not a real supervisor interface. Press Ctrl+C to stop early.

Each run replaces `data/traffic_data.csv` and `data/demo_trace.jsonl`. Its SQLite history is appended to `data/agent_memory.sqlite`. The analysis script prints the first ten readings, queue statistics and fault/review counts. It uses Python's built-in CSV support; no pandas or plotting installation is needed. It does not generate charts.

Sensor queues are observed before each action. The actual signal shows all-red clearance when switching. Post-step true queues in the CSV are simulation audit data, not additional inputs given to the AI. Blank sensor values mean missing data, not zero vehicles.

The lab scripts use the same decision logic as the original demo and save their outputs separately from `results`. The `source_sha256` in `results/summary.json` matches the current `src/traffic_agent.py`, so the saved results come from this exact source.

## Tests, experiment and demo

After installing (see above):

```bash
python -m unittest discover -s tests -v
python src/traffic_agent.py experiment
python src/traffic_agent.py demo
```

Training writes into `results` and replaces the saved evaluation files. The SQLite database keeps earlier demo runs; each has a distinct run ID. The most recent trace file is replaced on each demonstration.

To simulate a supervisor approving flagged decisions, use `python src/traffic_agent.py demo --approve`. This is a demonstration switch, not a real approval interface. Default behaviour withholds approval and uses the queue rule under the local interlocks. Before each decision the demo also counts faults (missing readings and rejected commands) in the last 12 events; three or more switch to a queue-rule safe mode and the learned action is not used. It also looks up precedent in `results/experience.sqlite`: if the other action reduced the queue more over 20 s in similar past situations, the step is sent for review.

## Report figures

The report's learning curve, Q-learning versus queue rule comparison and memory flowchart are drawn from the saved results. They use Matplotlib and SciPy, which `requirements.txt` installs:

```bash
python report_figures.py
python report_flowchart.py
```

They read `results` and do not retrain or rerun the agent.

## TensorBoard

The saved results can also be viewed in TensorBoard in a browser. It uses TensorBoard and tensorboardX, which `requirements.txt` installs:

```bash
python tensorboard_export.py
tensorboard --logdir runs --host 127.0.0.1
```

Open `http://localhost:6006`. Press Ctrl+C in the terminal to stop it. `--host 127.0.0.1` keeps it visible only on this computer.

- `training/alpha_*`: training queue, reward, exploration and switches for each learning rate. The HPARAMS tab shows each rate's validation score.
- `test/*`: mean queue on each of the 30 test seeds per scenario, for fixed timing, the queue rule, Q-learning and Q-learning with the memory safe mode.
- `demo/*`: the two saved 36-step demonstrations (approval withheld and approved): observed queues, Q-values, congestion probability, review flags, API failures and the reason for each decision (TEXT tab).

The export reads `results` and replaces `runs` each time. It does not retrain or rerun the agent, so TensorBoard shows saved results, not a live run.

## Files

| File | Purpose |
|---|---|
| `src/traffic_agent.py` | Simulator, Q-learning training, evaluation and memory demonstration |
| `config.json` | Training and simulation assumptions |
| `tests/test_agent.py` | Five behavioural checks |
| `results/q_table.npy` | Selected trained model |
| `results/state_visits.npy` | Training state visit counts |
| `results/training_log.csv` | Training snapshots every 50 episodes and final episode |
| `results/validation.csv` | Learning-rate comparison using validation seeds |
| `results/evaluation.csv` | All test episodes: 450 for the three main controllers plus 150 for Q-learning with the memory safe mode |
| `results/experience.sqlite` | Indexed experience base: 17,123 events from validation runs (situation, action, 20 s queue change), built by `experiment` |
| `results/penalty_ablation.csv` | Switch-penalty ablation: penalty 2 against 0, same training and test seeds |
| `results/sequence_flagged_step.png`, `results/operator_mockup.png` | Sequence diagram of one flagged step; design mock-up of a review screen (not implemented) |
| `results/training_seeds.csv` | Robustness check: the selected learning rate retrained on five training seeds and tested on the same seeds |
| `results/summary.json` | Aggregated results, source checksum and runtime version |
| `results/demo_trace.jsonl` | 36 decisions with observations, review outcomes and plant feedback |
| `results/agent_memory.sqlite` | Persistent event history (`events`) and review outcomes with triggers and reason (`operator_decisions`) |
| `report_figures.py` | Learning curve, paired Q-learning vs queue rule statistics, policy map, state visits, seed robustness, safe-mode, belief-timeline and demo-timeline charts |
| `report_flowchart.py` | Architecture diagram (report Figure 1.1) and memory-update flowchart (report Figure 1.4); uncertainty map (Figure 1.2); signal state machine and SQLite schema (Appendix D) |
| `results/architecture.png` | Agent architecture: policy, supervisor layer, memory and plant |
| `results/learning_curve.png` | Training queue length for the three learning rates |
| `results/qlearning_vs_queue_rule.csv` | Paired differences, 95% intervals, t-test, Holm and Wilcoxon p-values |
| `results/qlearning_vs_queue_rule.png` | Chart of those paired differences |
| `results/memory_flowchart.png` | Flowchart of one decision step and the memory updates |
| `results/policy_map.png` | Learned policy: phase the saved Q-table requests for each pair of queue bins (report Figure 2.1) |
| `results/sensor_fault_safe_mode.png` | Sensor-fault mean queue for the queue rule, Q-learning and the memory safe mode (report Figure 2.4) |
| `results/training_seed_robustness.png` | Q-learning minus queue rule for five training seeds (report Figure 2.5) |
| `results/state_visits.png` | Training visits per pair of queue bins (report Figure 2.6) |
| `results/uncertainty_map.png` | Each source of uncertainty, its effect, the design response and the remaining risk (report Figure 1.2) |
| `results/belief_timeline.png` | Observed queue and P(congestion) over the 36 demo steps (report Figure 1.3) |
| `results/demo_timeline.png` | Who decided each demo step: learned action, review, safe mode or fixed timing (report Figure 2.7) |
| `results/signal_state_machine.png`, `results/sqlite_schema.png` | Controller interlocks and SQLite tables (report Appendix D) |
| `tensorboard_export.py` | Writes the saved results as TensorBoard logs |
| `learning_demo.py` | Trains a new agent from an empty Q-table and shows it improving; `--plot` opens a live chart and policy map. Writes `data/learning_progress.csv` and `data/learning_snapshots.json`, never `results/` |
| `runs/` | TensorBoard logs. Not in the repository: run `python tensorboard_export.py` to create it from `results/` |

## Dataset and experimental scope

The dataset source is the synthetic generator `demand()` in the source file. Poisson arrival rates represent assumed vehicle counts per five-second interval. They are not observations from a real city. `config.json` and the saved seeds reproduce the arrivals. No external traffic dataset or SUMO simulation was used.

The environment aggregates north-south and east-west queues. It has no turning movements, pedestrians, vehicle acceleration, road geometry or neighbouring intersections. A single five-second clearance step simplifies amber and all-red operation. Timing values are teaching assumptions, not road engineering recommendations.

Evaluation compares the learned policy with fixed timing and a queue-based rule using identical arrivals and noise for each seed. It tests the policy with local interlocks. The `q_learning_safe` controller adds the short-term-memory safe mode without a human, so its test rows measure that part of the supervisor layer. The Bayesian risk flag and simulated supervisor gate run only in the separate demo, so the benchmark does not establish performance of the complete supervised agent.

The SQLite history is an audit knowledge base. The current implementation writes outcomes for later inspection and does not retrieve historical events to select actions. The short-term deque is read each step only to count recent faults for the safe mode. The separate experience base is searched each step by exact indexed key (with back-off to broader keys); it is not a vector database. The Q-table is separate learned long-term memory, loaded from disk for decisions. Do not claim a vector database, retrieval-augmented reasoning or online model retraining.

The selected policy was trained with seed 5023. Thirty test seeds measure traffic variability. `results/training_seeds.csv` repeats training with five seeds (5023, 8023, 11023, 14023, 17023); scenario standard deviations are 0.09 to 0.50 vehicles. The simulated assumptions have not been calibrated against real observations.
