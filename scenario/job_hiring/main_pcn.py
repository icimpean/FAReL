import pickle
import random

# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
# import cv2
import numpy as np
import wandb
from datetime import datetime
import uuid
import os

import torch
import argparse

from pytz import timezone

import sys
sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from agent.pcn.pcn import train, epsilon_metric, non_dominated, compute_hypervolume, run_episode, add_episode, \
    choose_commands, update_model, Transition, choose_action

from agent.pcn.logger import Logger
from fairness import SensitiveAttribute
from fairness.fairness_framework import FairnessFramework, ExtendedfMDP
from fairness.group import GroupNotion
from fairness.individual import IndividualNotion
from scenario.fraud_detection.MultiMAuS.simulator import parameters
from scenario.fraud_detection.MultiMAuS.simulator.transaction_model import TransactionModel
from scenario.fraud_detection.env import NUM_FRAUD_FEATURES, TransactionModelMDP, FraudFeature
from scenario.job_hiring.features import HiringFeature, Gender, ApplicantGenerator
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


def run_episode_fairness(env, model, desired_return, desired_horizon, max_return, results,
                         eval=False, normalise_state=False):
    transitions = []
    obs = env.reset()
    done = False
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
        results["main_reward"][-1].append(reward)
    results["main_reward"].append([])
    return transitions


def eval_(env, model, coverage_set, horizons, max_return, results, gamma=1., n=10, normalise_state=False):
    e_returns = np.empty((coverage_set.shape[0], n, coverage_set.shape[-1]))
    all_transitions = []
    for e_i, target_return, horizon in zip(np.arange(len(coverage_set)), coverage_set, horizons):
        n_transitions = []
        for n_i in range(n):
            transitions = run_episode_fairness(env, model, target_return, np.float32(horizon), max_return, results,
                                               eval=True, normalise_state=normalise_state)
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
    # since each state is a tuple with (compartment, events, prev_action), reorder obs
    # obs = zip(*obs)  # TODO disabled for hire/fraud, outputs singular state
    # obs = tuple([torch.tensor([o]).to(device) for o in obs])
    # print(obs)
    obs = torch.tensor(obs).to(device)
    # print("==>")
    # print(obs)

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

    return l, log_prob


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
               normalise_state=False
               ):
    step = 0
    if objectives == None:
        objectives = tuple([i for i in range(len(ref_point))])
    total_episodes = n_er_episodes
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger = Logger(logdir=logdir)
    n_checkpoints = 0

    results = {
        'notions': env.obj_names,
        'description': [],
        'reward': [],
        'cm': [],
        'overview': None,
        'main_reward': [[]],
        'transitions': [],
    }

    # fill buffer with random episodes
    experience_replay = []
    print("Experience replay...")
    for ep in range(n_er_episodes):
        transitions = []
        obs = env.reset()
        done = False
        while not done:
            curr_obs = env.normalise_state(obs) if normalise_state else obs
            action = np.random.randint(0, env.nA)
            n_obs, reward, done, info = env.step(action, scores=np.full(env.nA, fill_value=1/env.nA))
            next_obs = env.normalise_state(n_obs) if normalise_state else n_obs
            print("t=", step, ep, action, reward)
            results["main_reward"][-1].append(reward)

            if normalise_state:
                transitions.append(
                    Transition(curr_obs, action, np.float32(reward).copy(), next_obs, done))
            else:
                transitions.append(
                    Transition(curr_obs.to_array(), action, np.float32(reward).copy(), next_obs.to_array(), done))

            obs = n_obs
            step += 1
        # add episode in-place
        add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
        #
        results["transitions"].append(transitions)
        results["main_reward"].append([])
    with open(f'{logger.logdir}/fair_results_{n_checkpoints + 1}.pt', "wb") as f:
        pickle.dump(results, f)
    print("Training...")
    while step < total_steps:
        loss = []
        entropy = []
        for moupd in range(n_model_updates):
            l, lp = update_model_hire(model, opt, experience_replay, batch_size=batch_size)
            loss.append(l.detach().cpu().numpy())
            lp = lp.detach().cpu().numpy()
            ent = np.sum(-np.exp(lp) * lp)
            entropy.append(ent)

        desired_return, desired_horizon = choose_commands(experience_replay, n_er_episodes, objectives)

        # get all leaves, contain biggest elements, experience_replay got heapified in choose_commands
        leaves = np.array([(len(e[2]), e[2][0].reward) for e in experience_replay[len(experience_replay) // 2:]])
        e_lengths, e_returns = zip(*leaves)
        e_lengths, e_returns = np.array(e_lengths), np.array(e_returns)
        try:
            if len(experience_replay) == max_size:
                logger.put('train/leaves', e_returns, step, f'{e_returns.shape[-1]}d')
            # hv = hypervolume(e_returns[...,objectives]*-1)
            # hv_est = hv.compute(ref_point[objectives]*-1)
            # logger.put('train/hypervolume', hv_est, step, 'scalar')
            # wandb.log({'hypervolume': hv_est}, step=step)
        except ValueError:
            pass

        returns = []
        horizons = []
        for _ in range(n_step_episodes):
            transitions = run_episode_fairness(env, model, desired_return, desired_horizon, max_return, results,
                                               normalise_state=normalise_state)
            step += len(transitions)
            add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
            returns.append(transitions[0].reward)
            horizons.append(len(transitions))
            results["transitions"].append(transitions)
        with open(f'{logger.logdir}/fair_results_{n_checkpoints + 1}.pt', "wb") as f:
            pickle.dump(results, f)

        total_episodes += n_step_episodes
        logger.put('train/episode', total_episodes, step, 'scalar')
        logger.put('train/loss', np.mean(loss), step, 'scalar')
        logger.put('train/entropy', np.mean(entropy), step, 'scalar')
        logger.put('train/horizon/desired', desired_horizon, step, 'scalar')
        logger.put('train/horizon/distance', np.linalg.norm(np.mean(horizons) - desired_horizon), step, 'scalar')
        for o in range(len(desired_return)):
            logger.put(f'train/return/{o}/value', desired_horizon, step, 'scalar')
            logger.put(f'train/return/{o}/desired', np.mean(np.array(returns)[:, o]), step, 'scalar')
            logger.put(f'train/return/{o}/distance',
                       np.linalg.norm(np.mean(np.array(returns)[:, o]) - desired_return[o]), step, 'scalar')
        print(
            f'step {step} \t return {np.mean(returns, axis=0)}, ({np.std(returns, axis=0)}) \t loss {np.mean(loss):.3E}')

        # compute hypervolume of leaves
        valid_e_returns = e_returns[np.all(e_returns[:, objectives] >= ref_point[objectives,], axis=1)]
        hv = compute_hypervolume(np.expand_dims(valid_e_returns[:, objectives], 0), ref_point[objectives,])[0] if len(
            valid_e_returns) else 0

        wandb.log({
            'episode': total_episodes,
            'episode_steps': np.mean(horizons),
            'loss': np.mean(loss),
            'entropy': np.mean(entropy),
            'hypervolume': hv,
        }, step=step)

        if step >= (n_checkpoints + 1) * total_steps / 10:
            torch.save(model, f'{logger.logdir}/model_{n_checkpoints + 1}.pt')
            n_checkpoints += 1

            columns = env.obj_names if isinstance(env, ExtendedfMDP) else [f'o_{o}' for o in range(e_returns.shape[1])]

            coverage_set_table = wandb.Table(data=e_returns, columns=columns)

            # current coverage set
            _, e_i = non_dominated(e_returns[:, objectives], return_indexes=True)
            e_returns = e_returns[e_i]
            e_lengths = e_lengths[e_i]
            e_r, t_r = eval_(env, model, e_returns, e_lengths, max_return, results,
                             gamma=gamma, n=n_evaluations, normalise_state=normalise_state)
            # save raw evaluation returns
            logger.put(f'eval/returns/{n_checkpoints}', e_r, 0, f'{len(e_r)}d')
            # compute e-metric
            epsilon = epsilon_metric(e_r[..., objectives].mean(axis=1), e_returns[..., objectives])
            logger.put('eval/epsilon/max', epsilon.max(), step, 'scalar')
            logger.put('eval/epsilon/mean', epsilon.mean(), step, 'scalar')
            print('=' * 10, ' evaluation ', '=' * 10)
            for d, r in zip(e_returns, e_r):
                print('desired: ', d, '\t', 'return: ', r.mean(0))
            print(f'epsilon max/mean: {epsilon.max():.3f} \t {epsilon.mean():.3f}')
            print('=' * 22)

            nd_coverage_set_table = wandb.Table(data=e_returns * env.scale[None], columns=columns)
            nd_executions_table = wandb.Table(data=e_r.mean(axis=1) * env.scale[None], columns=columns)

            executions_transitions = wandb.Artifact(
                f'run-{wandb.run.id}-execution-transitions', type='transitions'
            )
            with executions_transitions.new_file('transitions.pkl', 'wb') as f:
                pickle.dump(t_r, f)

            wandb.log({
                'coverage_set': coverage_set_table,
                'nd_coverage_set': nd_coverage_set_table,
                'executions': nd_executions_table,
                'eps_max': epsilon.max(),
                'eps_mean': epsilon.mean(),
            }, step=step)
            wandb.run.log_artifact(executions_transitions)


if __name__ == '__main__':
    # import gym_covid
    # print(torch.cuda.is_available())

    import time
    t_start = time.time()

    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument('--objectives', default=[0, 1, 6], type=int, nargs='+',
                        help='index for reward (0), StatisticalParity (1), EqualOpportunity (2), '
                             'OverallAccuracyEquality (3), PredictiveParity (4), IndividualFairness (5), '
                             'ConsistencyScoreComplement (6)')
    parser.add_argument('--env', default='job', type=str, help='job or fraud')
    # parser.add_argument('--action', default='discrete', type=str, help='discrete, multidiscrete or continuous')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--steps', default=1e5, type=float, help='total timesteps')
    parser.add_argument('--batch', default=128, type=int, help='batch size')
    parser.add_argument('--model-updates', default=20, type=int,
                        help='number of times the model is updated at every training iteration')
    parser.add_argument('--top-episodes', default=10, type=int,
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
    parser.add_argument('--population', default='data/belgian_population.csv', type=str,
                        help='the name of the population file')
    # Fraud detection parameters
    # TODO
    # Fairness framework
    parser.add_argument('--window', default=100, type=int, help='fairness framework window')
    parser.add_argument('--fair_alpha', default=0.1, type=float, help='fairness framework alpha for similarity metric')

    #
    args = parser.parse_args()
    print(args)

    device = 'cpu'
    on_vsc = args.vsc == 1

    env_type = args.env
    # env_type = "fraud"  # TODO

    is_job_hiring = env_type == "job"

    n_evaluations = 10

    # args.top_episodes = 5  # TODO
    # args_n_episodes = 2
    # args.model_updates = 2

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if on_vsc:
        result_dir = "/data/brussel/104/vsc10437/Fairness/"
    else:
        result_dir = "../../../fairRLresults/"

    # Job hiring
    if is_job_hiring:
        logdir = f"{result_dir}/job_hiring/"
        team_size = args.team_size
        episode_length = args.episode_length
        diversity_weight = args.diversity_weight
        # Training environment
        applicant_generator = ApplicantGenerator(csv=args.population, seed=seed)
        env = JobHiringEnv(team_size=team_size, seed=seed, episode_length=episode_length,  # Required ep length for pcn
                           diversity_weight=diversity_weight, applicant_generator=applicant_generator)

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
        params['end_date'] = datetime(2016, 1, 7).replace(tzinfo=timezone('US/Pacific'))
        episode_length = np.sum(params["trans_per_year"]).astype(int) / 366 * (7)  # (90357) Used for min/max reward
        # print(episode_length)  # (31) 7653.188524590164, (14) 3456.27868852459, (7) 1728.139344262295
        # exit()

        transaction_model = TransactionModel(params, seed=seed)
        env = TransactionModelMDP(transaction_model, do_reward_shaping=True)

        # TODO: abstract parameters
        # Continents mapping from default parameters: {'EU': 0, 'AS': 1, 'NA': 2, 'AF': 3, 'OC': 4, 'SA': 5}
        sensitive_attribute = SensitiveAttribute(FraudFeature.continent, sensitive_values=1, other_values=0)

    #
    logdir += '/'.join([f'{k}_{v}' for k, v in vars(args).items()]) + '/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'
    os.makedirs(logdir, exist_ok=True)

    #
    main_obj = [0]

    all_group_notions = [GroupNotion.StatisticalParity, GroupNotion.EqualOpportunity,
                         GroupNotion.OverallAccuracyEquality, GroupNotion.PredictiveParity]
    all_individual_notions = [IndividualNotion.IndividualFairness, IndividualNotion.ConsistencyScoreComplement]

    fairness_framework = FairnessFramework([a for a in HiringActions], sensitive_attribute,
                                           individual_notions=all_individual_notions,
                                           group_notions=all_group_notions,
                                           get_individual=env.get_individual,
                                           similarity_metric=env.similarity_metric,
                                           alpha=args.fair_alpha,
                                           window=args.window)

    _num_notions = len(all_group_notions) + len(all_individual_notions)
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

    wandb.init(project=f'pcn-fair-{env_type}', entity='icimpean', config={k: v for k, v in vars(args).items()},
               dir=logdir)

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
               normalise_state=True
               )

    t_end = time.time()
    print(t_end - t_start, "seconds")
