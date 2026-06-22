# FAReL: Fairness-Aware Reinforcement Learning
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

FAReL introduces a multi-objective approach to reinforcement learning that balances traditional policy optimisation with fairness notions. 


This repository contains the official implementation of the framework presented in the paper:
> **Fairness-Aware Reinforcement
Learning (FAReL): A Framework for Transparent and Balanced Sequential Decision-Making.**  
> *Alexandra Cimpean, Nicole Orzan, Catholijn Jonker, Pieter Libin, and Ann Nowé*  
> https://arxiv.org/abs/2509.22232
---


### Directory Structure

Below is the layout of the repository's core files and directories:

```text
FAReL/
├── agent/                  # RL agent definitions
│   ├── dqn/                # Deep Q-Networks implementation
│   └── pcn/                # Pareto Conditioned Networks implementation
├── fairness/               # The fairness framework and history implementation
│   ├── group/              # Group fairness notions
│   └── individual/         # Individual fairness notions
├── loggers/                # Logging classes to write experimental data to files
├── scenario/               # Entry points for running experimental setups
│   ├── fraud_detection/    # Fraud detection environment
│   ├── job_hiring/         # Job hiring environment
│   ├── vis/                # Visualisation code
│   │   └── visualise.py    # Main file to visualise results from existing experiments
│   └── main_pcn_core.py    # Main script to run the experiments
└── requirements.txt        # Python dependencies
```

### Run experiments

The ```scenario/main_pcn_core.py``` file can be used to run experiments.
The script accepts the following parameters:

| Argument                          | Type | Default | Description                                                                                                                                |
|:----------------------------------| :--- | :--- |:-------------------------------------------------------------------------------------------------------------------------------------------|
| **PCN Setup**                     | | |                                                                                                                                            |
| `--lr`                            | `float` | `1e-3` | Learning rate for the network optimizer.                                                                                                   |
| `--steps`                         | `float` | `1e5` | Total environment timesteps for the experiment.                                                                                            |
| `--batch`                         | `int` | `256` | Mini-batch size used during model optimization.                                                                                            |
| `--model-updates`                 | `int` | `20` | Number of times the model is updated at every training iteration.                                                                          |
| `--top-episodes`                  | `int` | `50` | Top-n episodes used to compute target-return and horizon. Initially fills ER with n random episodes.                                       |
| `--n-episodes`                    | `int` | `10` | Number of episodes to run between each training iteration.                                                                                 |
| `--er-size`                       | `int` | `100` | Maximum capacity (in episodes) of the ER buffer.                                                                                           |
| `--threshold`                     | `float` | `0.02` | Crowding distance threshold before applying a performance penalty.                                                                         |
| `--noise`                         | `float` | `0.0` | Magnitude of noise applied to the target-return on batch-update.                                                                           |
| `--model`                         | `str` | `'densesmall'` | Selection for network architecture layout: dense(big &#124; small).                                                                        |
| **General Setup**                 | | |                                                                                                                                            |
| `--objectives`                    | `str` (list) | `['R', 'SP']` | Abbreviations of the fairness notions to optimise. Multiple values can be space-separated or colon-separated (e.g., `R:SP`).               |
| `--compute_objectives`            | `str` (list) | `['EO', 'OAE', 'PP', 'IF', 'CSC']` | Abbreviations of additional fairness notions to compute but not optimize. Colon-separable (e.g., `EO:OAE:PP`).                             |
| `--env`                           | `str` | `'job'` | Environment selection: `job` or `fraud`.                                                                                                   |
| `--seed`                          | `int` | `0` | Seed for the random number generator.                                                                                                      |
| `--vsc`                           | `int` | `0` | Execution environment: local execution (`0`) or VSC cluster (`1`).                                                                         |
| `--bias`                          | `int` | `0` | Which bias configuration to consider. Default `0` means no bias.                                                                           |
| `--ignore_sensitive`              | flag | `False` | Toggle to ignore sensitive attributes during run execution.                                                                                |
| **Job Hiring Parameters**         | | |                                                                                                                                            |
| `--team_size`                     | `int` | `20` | Maximum target team size to reach.                                                                                                         |
| `--episode_length`                | `int` | `100` | Maximum allowable timesteps per episode.                                                                                                   |
| `--diversity_weight`              | `int` | `0` | Diversity weight (acts as the complement of the skill weight).                                                                             |
| `--population`                    | `str` | `'belgian_population'` | File name of the target population dataset.                                                                                                |
| **Fraud Detection Parameters**    | | |                                                                                                                                            |
| `--n_transactions`                | `int` | `1000` | Number of transactions simulated per episode.                                                                                              |
| `--fraud_proportion`              | `float` | `0.0` | Proportion of fraudulent transactions to genuine. `0` defaults to standard MultiMAuS parameters.                                           |
| **Fairness Framework**            | | |                                                                                                                                            |
| `--window`                        | `int` | `100` | Size of the sliding window for the fairness framework.                                                                                     |
| `--no_window`                     | `int` | `0` | Set to `1` to use the full tracking history instead of a sliding window.                                                                   |
| `--discount_history`              | flag | `False` | Toggle to use a discounted history instead of the default sliding window implementation.                                                   |
| `--discount_factor`               | `float` | `1.0` | Fairness framework history discount factor.                                                                                                |
| `--discount_threshold`            | `float` | `1e-5` | History retention threshold cut-off for discounted history.                                                                                |
| `--discount_delay`                | `int` | `5` | Timesteps needed for a fairness notion to not fluctuate more than `discount_threshold` before dropping old interactions.                   |
| `--min_window`                    | `int` | `100` | Minimum allowable window size for discounted history.                                                                                      |
| `--no_individual`                 | `int` | `0` | Set to `1` to completely disable individual fairness evaluations.                                                                          |
| `--nearest_neighbours`            | `int` | `5` | Number of neighbors evaluated for the CSC individual fairness notion.                                                                      |
| `--fair_alpha`                    | `float` | `0.1` | Alpha weighting parameter utilized within the fairness framework similarity metric.                                                        |
| `--distance_metrics`              | `str` (list) | `['braycurtis']` | Distance metric mapping per individual fairness notion (e.g., `braycurtis:HEOM`). Passed in order of objectives, then computed objectives. |
| `--combined_sensitive_attributes` | `int` | `0` | Set to `1` to evaluate fairness using an intersectional combination of multiple sensitive attributes.                                      |
| **Logging & Tracking**            | | |                                                                                                                                            |
| `--wandb`                         | `int` | `1` | Enforces local-only logging overrides.                                                     |
| `--log_dir`                       | `str` | `'new_experiment'` | Target output directory where results are saved.                                                      |
| `--log_compact`                   | flag | `False` | Generates highly compressed/compact log variants to save storage space.                                                                    |
| `--log_coverage_set_only`         | flag | `False` | Toggles storage constraints to isolate and record only the coverage set data.                                                     |
---

