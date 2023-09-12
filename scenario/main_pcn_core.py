import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import os

import argparse

from pytz import timezone

import sys

sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from agent.pcn.pcn_core import epsilon_metric, non_dominated, compute_hypervolume, add_episode, \
    choose_commands, Transition

from agent.pcn.logger import Logger
from fairness import SensitiveAttribute, CombinedSensitiveAttribute
from fairness.fairness_framework import FairnessFramework, ExtendedfMDP
from fairness.group import GroupNotion
from fairness.individual import IndividualNotion
from loggers.logger import AgentLogger, LeavesLogger, TrainingPCNLogger, EvalLogger
from scenario.fraud_detection.MultiMAuS.simulator import parameters
from scenario.fraud_detection.MultiMAuS.simulator.transaction_model import TransactionModel
from scenario.fraud_detection.env import NUM_FRAUD_FEATURES, TransactionModelMDP, FraudFeature
from scenario.job_hiring.features import HiringFeature, Gender, ApplicantGenerator, Nationality
from scenario.job_hiring.env import HiringActions, JobHiringEnv, NUM_JOB_HIRING_FEATURES

ss_emb = {
    'job': {
        'small': nn.Sequential(
            # nn.Flatten(),
            nn.Linear(NUM_JOB_HIRING_FEATURES, 64),
            nn.Sigmoid()
        ),
        'big': nn.Sequential(
            # nn.Flatten(),
            nn.Linear(NUM_JOB_HIRING_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        ),
    },
    'fraud': {
        'small': nn.Sequential(
            # nn.Flatten(),
            nn.Linear(NUM_FRAUD_FEATURES, 64),
            nn.Sigmoid()
        ),
        'big': nn.Sequential(
            # nn.Flatten(),
            nn.Linear(NUM_FRAUD_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        ),
    }
}

se_emb = {
    'small': nn.Sequential(
        nn.Linear(1, 64),
        nn.Sigmoid()
    ),
    'big': nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.Sigmoid()
    )
}

sa_emb = {
    'small': nn.Sequential(
        nn.Linear(3, 64),
        nn.Sigmoid()
    ),
    'big': nn.Sequential(
        nn.Linear(3, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.Sigmoid()
    )
}


class Model(nn.Module):
    def __init__(self,
                 nA,
                 scaling_factor,
                 objectives,
                 ss_emb):
        super(Model, self).__init__()

        self.scaling_factor = scaling_factor[:, objectives + (len(scaling_factor) - 1,)]
        self.objectives = objectives
        self.ss_emb = ss_emb
        self.s_emb = nn.Sequential(
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        self.c_emb = nn.Sequential(nn.Linear(self.scaling_factor.shape[-1], 64),
                                   nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, nA))

    def forward(self, state, desired_return, desired_horizon):
        desired_return = desired_return[:, self.objectives]
        c = torch.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c * self.scaling_factor
        s = self.ss_emb(state.float())
        s = self.s_emb(s)
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        sc = s * c
        log_prob = self.fc(sc)
        return log_prob


class DiscreteHead(nn.Module):
    def __init__(self, base):
        super(DiscreteHead, self).__init__()
        self.base = base

    def forward(self, state, desired_return, desired_horizon):
        x = self.base(state, desired_return, desired_horizon)
        x = F.log_softmax(x, 1)
        return x


def run_episode_fairness(env, model, desired_return, desired_horizon, max_return, agent_logger, current_ep, current_t,
                         eval=False, normalise_state=False, eval_axes=False):
    curr_t = time.time()
    transitions = []
    obs = env.reset()
    done = False
    t = current_t
    log_entries = []
    if eval and eval_axes:
        path = agent_logger.path_eval_axes
        status = "eval_axes"
    elif eval:
        path = agent_logger.path_eval
        status = "eval"
    else:
        path = agent_logger.path_train
        status = "train"
    while not done:
        curr_obs = env.normalise_state(obs) if normalise_state else obs
        action, scores = choose_action_hire(model, curr_obs if normalise_state else curr_obs.to_array(),
                                            desired_return,
                                            desired_horizon, eval=eval, return_probs=True)
        n_obs, reward, done, info = env.step(action, scores)
        next_obs = env.normalise_state(n_obs) if normalise_state else n_obs

        transitions.append(Transition(
            observation=curr_obs if normalise_state else curr_obs.to_array(),
            action=action,
            reward=np.float32(reward).copy(),
            next_observation=next_obs if normalise_state else next_obs.to_array(),
            terminal=done
        ))

        obs = n_obs
        # clip desired return, to return-upper-bound,
        # to avoid negative returns giving impossible desired returns
        desired_return = np.clip(desired_return - reward, None, max_return, dtype=np.float32)
        # clip desired horizon to avoid negative horizons
        desired_horizon = np.float32(max(desired_horizon - 1, 1.))
        #
        next_t = time.time()
        if not eval_axes:
            log_entries.append(
                agent_logger.create_entry(current_ep, t, obs, action, reward, done, info, next_t - curr_t,
                                          status))
        curr_t = next_t
        t += 1
    agent_logger.write_data(log_entries, path)
    return transitions


def eval_(env, model, coverage_set, horizons, max_return, agent_logger, current_ep, current_t, gamma=1., n=10,
          normalise_state=False, eval_axes=False):
    e_returns = np.empty((coverage_set.shape[0], n, coverage_set.shape[-1]))
    all_transitions = []
    for e_i, target_return, horizon in zip(np.arange(len(coverage_set)), coverage_set, horizons):
        n_transitions = []
        for n_i in range(n):
            transitions = run_episode_fairness(env, model, target_return, np.float32(horizon), max_return, agent_logger,
                                               current_ep, current_t, eval=True, normalise_state=normalise_state,
                                               eval_axes=eval_axes)
            # compute return
            for i in reversed(range(len(transitions) - 1)):
                transitions[i].reward += gamma * transitions[i + 1].reward
            e_returns[e_i, n_i] = transitions[0].reward
            n_transitions.append(transitions)
        all_transitions.append(n_transitions)

    return e_returns, all_transitions


def update_model_hire(model, opt, experience_replay, batch_size, noise=0.):
    batch = []
    # randomly choose episodes from experience buffer
    s_i = np.random.choice(np.arange(len(experience_replay)), size=batch_size, replace=True)
    for i in s_i:
        # episode is tuple (return, transitions)
        ep = experience_replay[i][2]
        # choose random timestep from episode,
        # use it's return and leftover timesteps as desired return and horizon
        t = np.random.randint(0, len(ep))
        # reward contains return until end of episode
        s_t, a_t, r_t, h_t = ep[t].observation, ep[t].action, np.float32(ep[t].reward), np.float32(len(ep) - t)
        batch.append((s_t, a_t, r_t, h_t))

    obs, actions, desired_return, desired_horizon = zip(*batch)
    obs = torch.tensor(obs).to(device)

    # TODO TEST add noise to the desired return
    desired_return = torch.tensor(desired_return).to(device)
    desired_return = desired_return + noise * torch.normal(0, 1, size=desired_return.shape,
                                                           device=desired_return.device)
    log_prob = model(obs,
                     desired_return,
                     torch.tensor(desired_horizon).unsqueeze(1).to(device))
    opt.zero_grad()

    # check if actions are continuous
    # TODO hacky
    if model.__class__.__name__ == 'ContinuousHead':
        l = F.mse_loss(log_prob, torch.tensor(actions))
    else:
        # one-hot of action for CE loss
        actions = torch.tensor(actions).long().to(device)
        actions = F.one_hot(actions, num_classes=log_prob.shape[-1])
        # cross-entropy loss
        l = torch.sum(-actions * log_prob, -1).sum(-1)
    l = l.mean()
    l.backward()
    opt.step()

    return l.detach().cpu().numpy(), log_prob.detach().cpu().numpy()


def choose_action_hire(model, obs, desired_return, desired_horizon, eval=False, return_probs=False):
    # if observation is not a simple np.array, convert individual arrays to tensors
    obs = [torch.tensor([o]).to(device) for o in obs] if type(obs) == tuple else torch.tensor([obs]).to(device)
    log_probs = model(obs,
                      torch.tensor([desired_return]).to(device),
                      torch.tensor([desired_horizon]).unsqueeze(1).to(device))
    log_probs = log_probs.detach().cpu().numpy()[0]
    # check if actions are continuous
    # TODO hacky
    if model.__class__.__name__ == 'ContinuousHead':
        action = log_probs
        # add some noise for randomness
        if not eval:
            action = np.clip(action + np.random.normal(0, 0.1, size=action.shape).astype(np.float32), 0, 1)
    else:
        # if evaluating: act greedily
        if eval:
            action = np.argmax(log_probs, axis=-1)
            if return_probs:
                return action, log_probs
            else:
                return action

        if log_probs.ndim == 1:
            log_probs = np.nan_to_num(log_probs)
            if log_probs.sum() == 0:
                log_probs = np.full_like(log_probs, 1 / len(log_probs))
            log_probs = np.exp(log_probs)
            if log_probs.sum() != 1:
                log_probs = log_probs / log_probs.sum()
            action = np.random.choice(np.arange(len(log_probs)), p=log_probs)
        elif log_probs.ndim == 2:
            action = np.array(list([np.random.choice(np.arange(len(lp)), p=np.exp(lp)) for lp in log_probs]))
    if return_probs:
        return action, log_probs
    else:
        return action


def train_fair(env,
               model,
               learning_rate=1e-2,
               batch_size=1024,
               total_steps=1e4,
               n_model_updates=100,
               n_step_episodes=10,
               n_er_episodes=10,
               gamma=1.,
               max_return=250.,
               max_size=500,
               ref_point=np.array([0, 0]),
               threshold=0.2,
               noise=0.0,
               objectives=None,
               n_evaluations=10,
               logdir='runs/',
               normalise_state=False,
               use_wandb=True
               ):
    step = 0
    if objectives == None:
        objectives = tuple([i for i in range(len(ref_point))])
    total_episodes = n_er_episodes
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if use_wandb:
        logger = Logger(logdir=logdir)
    agent_logger = AgentLogger(f"{logdir}/agent_log_e_replay.csv", f"{logdir}/agent_log_train.csv",
                               f"{logdir}/agent_log_eval.csv", f"{logdir}/agent_log_eval_axes.csv")
    # leaves_logger = LeavesLogger(
    #     objective_names=env.obj_names if isinstance(env, ExtendedfMDP) else [f'o_{o}' for o in objectives])
    all_obj = [i for i in range(len(ref_point))]
    pcn_logger = TrainingPCNLogger(objectives=all_obj)
    eval_logger = EvalLogger(objectives=all_obj)

    # TODO: replace by list of timesteps and actions to recreate interactions?
    # agent_logger.create_file(agent_logger.path_eval_axes)
    agent_logger.create_file(agent_logger.path_eval)
    agent_logger.create_file(agent_logger.path_train)
    agent_logger.create_file(agent_logger.path_experience)

    # leaves_logger.create_file(f"{logdir}/leaves_log.csv")
    pcn_logger.create_file(f"{logdir}/pcn_log.csv")
    eval_logger.create_file(f"{logdir}/eval_log.csv")
    n_checkpoints = 0

    # fill buffer with random episodes
    experience_replay = []
    print("Experience replay...")

    for ep in range(n_er_episodes):
        curr_t = time.time()
        log_entries = []
        transitions = []
        obs = env.reset()
        done = False
        while not done:
            curr_obs = env.normalise_state(obs) if normalise_state else obs
            action = np.random.randint(0, env.nA)
            n_obs, reward, done, info = env.step(action, scores=np.full(env.nA, fill_value=1 / env.nA))
            next_obs = env.normalise_state(n_obs) if normalise_state else n_obs
            # TODO
            if step % 20 == 0:
                print("t=", step, ep, action, reward)

            transitions.append(
                Transition(curr_obs if normalise_state else curr_obs.to_array(), action, np.float32(reward).copy(),
                           next_obs if normalise_state else next_obs.to_array(), done))
            next_t = time.time()

            log_entries.append(agent_logger.create_entry(ep, step, obs, action, reward, done, info, next_t - curr_t,
                                                         status="e_replay"))
            curr_t = next_t

            obs = n_obs
            step += 1
        # add episode in-place
        print(f"Store episode {ep}, t {step}")
        add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
        agent_logger.write_data(log_entries, agent_logger.path_experience)

    del log_entries

    print("Training...")
    update_num = 0

    while step < total_steps:
        print("loop", update_num)

        loss = []
        entropy = []
        for moupd in range(n_model_updates):
            l, lp = update_model_hire(model, opt, experience_replay, batch_size=batch_size)
            loss.append(l)
            lp = lp
            ent = np.sum(-np.exp(lp) * lp)
            entropy.append(ent)
        print("model updates", update_num)

        desired_return, desired_horizon = choose_commands(experience_replay, n_er_episodes, objectives)

        # get all leaves, contain biggest elements, experience_replay got heapified in choose_commands
        # print([(len(e[2]), e[2][0].reward) for e in experience_replay[len(experience_replay) // 2:]])
        # leaves = np.array([(len(e[2]), e[2][0].reward) for e in experience_replay[len(experience_replay) // 2:]])
        # e_lengths, e_returns = zip(*leaves)
        # print([(len(e[2]), e[2][0].reward) for e in experience_replay[len(experience_replay) // 2:]])
        # leaves = np.array([(len(e[2]), e[2][0].reward) for e in experience_replay[len(experience_replay) // 2:]])
        e_lengths, e_returns = [(len(e[2])) for e in experience_replay[len(experience_replay) // 2:]], [(e[2][0].reward)
                                                                                                        for e in
                                                                                                        experience_replay[
                                                                                                        len(experience_replay) // 2:]]
        e_lengths, e_returns = np.array(e_lengths), np.array(e_returns)
        try:
            if len(experience_replay) == max_size:
                if use_wandb:
                    logger.put('train/leaves', e_returns, step, f'{e_returns.shape[-1]}d')
                else:
                    leaves = []
                    # for er in e_returns:
                    #     leaves.append(leaves_logger.create_entry(ep, step, er))
                    # leaves_logger.write_data(leaves)
            # hv = hypervolume(e_returns[...,objectives]*-1)
            # hv_est = hv.compute(ref_point[objectives]*-1)
            # logger.put('train/hypervolume', hv_est, step, 'scalar')
            # wandb.log({'hypervolume': hv_est}, step=step)
        except ValueError:
            pass

        returns = []
        horizons = []
        for _ in range(n_step_episodes):
            transitions = run_episode_fairness(env, model, desired_return, desired_horizon, max_return, agent_logger,
                                               normalise_state=normalise_state, current_t=step, current_ep=ep)
            step += len(transitions)
            ep += 1
            add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
            returns.append(transitions[0].reward)
            horizons.append(len(transitions))

        print(
            f'step {step} \t return {np.mean(returns, axis=0)}, ({np.std(returns, axis=0)}) \t loss {np.mean(loss):.3E}')

        # compute hypervolume of leaves
        valid_e_returns = e_returns[np.all(e_returns[:, objectives] >= ref_point[objectives,], axis=1)]
        hv = compute_hypervolume(np.expand_dims(valid_e_returns[:, objectives], 0), ref_point[objectives,])[0] if len(
            valid_e_returns) and len(objectives) > 1 else 0

        # current coverage set
        nd_coverage_set, e_i = non_dominated(e_returns[:, objectives], return_indexes=True)

        entry = pcn_logger.create_entry(ep, step, np.mean(loss), np.mean(entropy), desired_horizon,
                                        np.linalg.norm(np.mean(horizons) - desired_horizon), np.mean(horizons), hv,
                                        e_returns, nd_coverage_set,
                                        np.mean(np.array(returns), axis=0), desired_return,
                                        [np.linalg.norm(np.mean(np.array(returns)[:, o]) - desired_return[o]) for o in
                                         range(len(desired_return))])
        pcn_logger.write_data(entry)

        if step >= (n_checkpoints + 1) * total_steps / 10:
            if not no_save:  # torch.save gives errors when reached with memory profilers runs
                torch.save(model, f'{logdir}/model_{n_checkpoints + 1}.pt')
            n_checkpoints += 1

            columns = env.obj_names if isinstance(env, ExtendedfMDP) else [f'o_{o}' for o in range(e_returns.shape[1])]

            # # current coverage set
            # _, e_i = non_dominated(e_returns[:, objectives], return_indexes=True)
            e_returns = e_returns[e_i]
            e_lengths = e_lengths[e_i]
            e_r, t_r = eval_(env, model, e_returns, e_lengths, max_return, agent_logger, ep, step,
                             gamma=gamma, n=n_evaluations, normalise_state=normalise_state)

            # compute e-metric
            epsilon = epsilon_metric(e_r[..., objectives].mean(axis=1), e_returns[..., objectives])
            print('=' * 10, ' evaluation ', '=' * 10)
            for d, r in zip(e_returns, e_r):
                print('desired: ', d, '\t', 'return: ', r.mean(0))
            print(f'epsilon max/mean: {epsilon.max():.3f} \t {epsilon.mean():.3f}')
            print('=' * 22)

            entries = []
            for d, r in zip(e_returns, e_r):
                entry = eval_logger.create_entry(ep, step, epsilon.max(), epsilon.mean(), d, r.mean(0), "eval")
                entries.append(entry)
            eval_logger.write_data(entries)

        update_num += 1


if __name__ == '__main__':
    # import gym_covid
    # print(torch.cuda.is_available())
    # import objgraph
    import time

    t_start = time.time()

    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument('--objectives', default=[0, 1], type=int, nargs='+',
                        help='index for reward (0), StatisticalParity (1), EqualOpportunity (2), '
                             'OverallAccuracyEquality (3), PredictiveParity (4), '
                             'IndividualFairness (5), ConsistencyScoreComplement (6),'
                             'WeaklyMeritocratic (7), Gini (8), Entropy (9)')  # TODO connect/implement 7+
    parser.add_argument('--single_objective', default=-1, type=int, help="Use a single objective to train on")
    parser.add_argument('--env', default='job', type=str, help='job or fraud')
    # parser.add_argument('--action', default='discrete', type=str, help='discrete, multidiscrete or continuous')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--steps', default=1e5, type=float, help='total timesteps')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--model-updates', default=20, type=int,
                        help='number of times the model is updated at every training iteration')
    parser.add_argument('--top-episodes', default=50, type=int,
                        help='top-n episodes used to compute target-return and horizon. \
              Initially fill ER with n random episodes')
    parser.add_argument('--n-episodes', default=10, type=int,
                        help='number of episodes to run between each training iteration')
    parser.add_argument('--er-size', default=100, type=int,
                        help='max size (in episodes) of the ER buffer')
    parser.add_argument('--threshold', default=0.02, type=float, help='crowding distance threshold before penalty')
    parser.add_argument('--noise', default=0.0, type=float, help='noise applied on target-return on batch-update')
    parser.add_argument('--model', default='densesmall', type=str, help='dense(big|small)')
    #
    parser.add_argument('--seed', default=0, type=int, help='seed for rng')
    parser.add_argument('--vsc', default=0, type=int, help='running on local (0) or VSC cluster (1)')

    # Job hiring parameters
    parser.add_argument('--team_size', default=20, type=int, help='maximum team size to reach')
    parser.add_argument('--episode_length', default=100, type=int, help='maximum episode length')
    parser.add_argument('--diversity_weight', default=0, type=int, help='diversity weight, complement of skill weight')
    parser.add_argument('--population', default='belgian_population', type=str,
                        help='the name of the population file')
    # Fraud detection parameters
    # TODO
    parser.add_argument('--n_transactions', default=1000, type=int, help='number of transactions per episode')
    parser.add_argument('--fraud_proportion', default=0, type=float,
                        help='proportion of fraudulent transactions to genuine. 0 defaults to default MultiMAuS parameters')
    # Fairness framework
    parser.add_argument('--window', default=100, type=int, help='fairness framework window')
    parser.add_argument('--discount_history', default=0, type=int,
                        help='use a discounted history or sliding window implementation')
    parser.add_argument('--discount_factor', default=1.0, type=float,
                        help='fairness framework discount factor for history')
    parser.add_argument('--discount_threshold', default=1e-5, type=float,
                        help='fairness framework discount threshold for history')
    parser.add_argument('--fair_alpha', default=0.1, type=float, help='fairness framework alpha for similarity metric')
    parser.add_argument('--wandb', default=1, type=int,
                        help="(Ignored, overrides to 0) use wandb for loggers or save local only")
    parser.add_argument('--no_window', default=0, type=int, help="Use the full history instead of a window")
    parser.add_argument('--no_individual', default=0, type=int, help="No individual fairness notions")
    parser.add_argument('--distance_metrics', default=[], type=str, nargs='+',
                        help='The distance metric to use for every individual fairness notion specified')
    #
    parser.add_argument('--combined_sensitive_attributes', default=0, type=int,
                        help='Use a combination of sensitive attributes to compute fairness notions')
    #
    parser.add_argument('--log_dir', default='new_experiment', type=str, help="Directory where to store results")

    args = parser.parse_args()
    no_save = False
    args.wandb = 0

    # args.top_episodes = 5  # TODO
    # args.n_episodes = 5
    # args.er_size = 20
    # args.model_updates = 10
    # args.steps = 10000
    # # args.team_size = 100
    # # args.episode_length = args.team_size * 10
    # args.window = 100
    # # args.window = None
    # # args.no_individual = 1
    # # args.default_objectives = 1
    # # args.no_window = 0
    # # #
    # #
    # # # args.single_objective = 0
    # # args.env = "fraud"
    # # args.seed = 1
    # # no_save = True  # TODO
    # # args.fraud_proportion = 0#.20
    # args.log_dir = "./experiment/individual_fair/"
    # # args.vsc = 2
    # # args.discount_history = 1
    # # args.discount_factor = 1.0
    # # args.discount_threshold = 1e-4
    # args.combined_sensitive_attributes = 2
    #
    # args.objectives = [0, 1, 5, 6, 5, 6, 5, 6]
    # args.distance_metrics = ["braycurtis", "braycurtis", "HMOM", "HMOM", "HEOM", "HEOM"]
    # # args.objectives = [0, 1, 5, 5, 5]
    # # args.distance_metrics = ["braycurtis", "HMOM", "HEOM"]

    print(args)

    arg_use_wandb = args.wandb == 1
    device = 'cpu'
    on_vsc = args.vsc == 1

    env_type = args.env
    is_job_hiring = env_type == "job"
    n_evaluations = 10

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.no_window:
        args.window = None

    if args.single_objective != -1:
        args.objectives = [args.single_objective]
        print("Single objective:", args.objectives)

    if on_vsc:
        result_dir = "/data/brussel/104/vsc10437/Fairness/"
    else:
        result_dir = "../../fairRLresults/"

    # Job hiring
    if is_job_hiring:
        logdir = f"{result_dir}/job_hiring/"
        team_size = args.team_size
        episode_length = args.episode_length
        diversity_weight = args.diversity_weight
        # Training environment
        population_file = f'./scenario/job_hiring/data/{args.population}.csv'
        if not on_vsc and args.vsc != 2:
            population_file = "." + population_file
        applicant_generator = ApplicantGenerator(csv=population_file, seed=seed)
        env = JobHiringEnv(team_size=team_size, seed=seed, episode_length=episode_length,  # Required ep length for pcn
                           diversity_weight=diversity_weight, applicant_generator=applicant_generator)

        if args.combined_sensitive_attributes == 1:
            sensitive_attribute = CombinedSensitiveAttribute([HiringFeature.gender, HiringFeature.nationality],
                                                             sensitive_values=[Gender.female, Nationality.foreign],
                                                             other_values=[Gender.male, Nationality.belgian])
        elif args.combined_sensitive_attributes == 2:
            sensitive_attribute = [SensitiveAttribute(HiringFeature.gender, sensitive_values=Gender.female,
                                                      other_values=Gender.male),
                                   SensitiveAttribute(HiringFeature.nationality, sensitive_values=Nationality.foreign,
                                                      other_values=Nationality.belgian)]
        else:
            sensitive_attribute = SensitiveAttribute(HiringFeature.gender, sensitive_values=Gender.female,
                                                     other_values=Gender.male)  # TODO: abstract parameters

    # Fraud
    else:
        logdir = f"{result_dir}/fraud_detection/"
        # the parameters for the simulation
        params = parameters.get_default_parameters()  # TODO: abstract parameters
        params['seed'] = seed
        params['init_satisfaction'] = 0.9
        params['stay_prob'] = [0.9, 0.6]
        params['num_customers'] = 100
        params['num_fraudsters'] = 10
        # params['end_date'] = datetime(2016, 12, 31).replace(tzinfo=timezone('US/Pacific'))
        # params['end_date'] = datetime(2016, 3, 31).replace(tzinfo=timezone('US/Pacific'))
        # # TODO (90357 yearly) Used for min/max reward
        # episode_length = np.sum(params["trans_per_year"]).astype(int) / 366 * (31 + 29 + 31)
        # 1 week = +- 1728 transactions
        num_transactions = args.n_transactions  # 1000
        params['end_date'] = datetime(2016, 1, 7).replace(tzinfo=timezone('US/Pacific'))
        # episode_length = np.sum(params["trans_per_year"]).astype(int) / 366 * (7)  # (90357) Used for min/max reward
        episode_length = num_transactions
        if args.fraud_proportion != 0:
            curr_sum = np.sum(params['trans_per_year'])
            params['trans_per_year'] = np.array([curr_sum * (1 - args.fraud_proportion),
                                                 curr_sum * args.fraud_proportion])

        transaction_model = TransactionModel(params, seed=seed)
        env = TransactionModelMDP(transaction_model, do_reward_shaping=True, num_transactions=num_transactions)

        # TODO: abstract parameters
        # Continents mapping from default parameters: {'EU': 0, 'AS': 1, 'NA': 2, 'AF': 3, 'OC': 4, 'SA': 5}
        # sensitive_attribute = SensitiveAttribute(FraudFeature.continent, sensitive_values=1, other_values=0)
        # NA vs EU instead of AS vs EU to increase population size in both
        if args.combined_sensitive_attributes == 1:
            sensitive_attribute = CombinedSensitiveAttribute([FraudFeature.continent, FraudFeature.merchant_id],
                                                             sensitive_values=[2, 6],
                                                             other_values=[0, None])
        elif args.combined_sensitive_attributes == 2:
            sensitive_attribute = [SensitiveAttribute(FraudFeature.continent, sensitive_values=2,
                                                      other_values=0),
                                   SensitiveAttribute(FraudFeature.merchant_id, sensitive_values=6,
                                                      other_values=None)]
        else:
            sensitive_attribute = SensitiveAttribute(FraudFeature.continent, sensitive_values=2, other_values=0)

    #
    logdir += args.log_dir + "/"
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S/')
    os.makedirs(logdir, exist_ok=True)

    #
    main_obj = [0]

    all_group_notions = [GroupNotion.StatisticalParity, GroupNotion.EqualOpportunity,
                         GroupNotion.OverallAccuracyEquality, GroupNotion.PredictiveParity]

    _ind_notions_mapping = {
        5: IndividualNotion.IndividualFairness,
        # 6: IndividualNotion.ConsistencyScoreComplement,  # TODO: replace/remove
        6: IndividualNotion.ConsistencyScoreComplementOnline,
    }

    all_individual_notions = [_ind_notions_mapping[o] for o in args.objectives if o >= 5]
    if args.no_individual:
        all_individual_notions = []

    use_discount_history = args.discount_history != 0
    discount_factor = args.discount_factor if use_discount_history else None
    discount_threshold = args.discount_threshold if use_discount_history else None
    fairness_framework = FairnessFramework([a for a in HiringActions], sensitive_attribute,
                                           individual_notions=all_individual_notions,
                                           group_notions=all_group_notions,
                                           get_individual=env.get_individual,
                                           similarity_metric=env.similarity_metric,
                                           distance_metrics=args.distance_metrics,
                                           alpha=args.fair_alpha,
                                           window=args.window,
                                           discount_factor=discount_factor,
                                           discount_threshold=discount_threshold,
                                           store_interactions=False, has_individual_fairness=not args.no_individual)

    # TODO:  #notions = #group notions + #individual notions with specific similarity distance
    # TODO: max reward still ok with new metrics/group divisions
    _num_group_notions = (len(sensitive_attribute) if args.combined_sensitive_attributes >= 2 else 1) * len(all_group_notions)
    _num_notions = _num_group_notions + len(all_individual_notions)
    max_reward = episode_length * 1
    scale = np.array([1] + [1] * _num_notions)
    ref_point = np.array([-max_reward] + [-episode_length] * _num_notions)
    scaling_factor = torch.tensor([[1.0] + ([1] * _num_notions) + [0.1]]).to(device)
    max_return = np.array([max_reward] + [0] * _num_notions) / scale

    input_shape = env.input_shape
    actions = env.actions

    # Extend the environment with fairness framework
    env = ExtendedfMDP(env, fairness_framework)

    env.nA = len(actions)
    env.scale = scale

    kw = "small" if args.model == "densesmall" else "big"
    ss, se, sa = ss_emb[env_type][kw], se_emb[kw], sa_emb[kw]

    model = Model(env.nA, scaling_factor, tuple(args.objectives), ss).to(device)
    model = DiscreteHead(model)

    train_fair(env,
               model,
               learning_rate=args.lr,
               batch_size=args.batch,
               total_steps=args.steps,
               n_model_updates=args.model_updates,
               n_er_episodes=args.top_episodes,
               n_step_episodes=args.n_episodes,
               max_size=args.er_size,
               max_return=max_return,
               threshold=args.threshold,
               ref_point=ref_point,
               noise=args.noise,
               n_evaluations=n_evaluations,
               objectives=tuple(args.objectives),
               logdir=logdir,
               normalise_state=True,
               use_wandb=arg_use_wandb
               )

    t_end = time.time()
    print(t_end - t_start, "seconds")
