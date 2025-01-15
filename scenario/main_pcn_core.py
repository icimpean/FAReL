import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from agent.pcn.pcn_core import epsilon_metric, non_dominated, compute_hypervolume, add_episode, \
    choose_commands, Transition
from agent.pcn.logger import Logger
from create_fair_env import *
from fairness.fairness_framework import ExtendedfMDP
from loggers.logger import AgentLogger, LeavesLogger, TrainingPCNLogger, EvalLogger, DiscountHistoryLogger
from scenario.fraud_detection.env import NUM_FRAUD_FEATURES
from scenario.job_hiring.env import NUM_JOB_HIRING_FEATURES


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


def run_episode_fairness(env, model, desired_return, desired_horizon, max_return, agent_logger, discount_history_logger,
                         current_ep, current_t, eval=False, normalise_state=False, eval_axes=False,
                         log_compact=False, log_coverage_set_only=False):
    curr_t = time.time()
    transitions = []
    obs = env.reset()
    done = False
    t = current_t
    log_entries = []
    history_entries = []

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
        if not eval_axes or log_coverage_set_only or not (eval and log_compact):
            log_entries.append(
                agent_logger.create_entry(current_ep, t, obs, action, reward, done, info, next_t - curr_t,
                                          status))
        if discount_history_logger:
            history_entries.append(discount_history_logger.create_entry(current_ep, t,
                                                                        env.fairness_framework.history.get_size(),
                                                                        env.fairness_framework.history.difference,
                                                                        env.fairness_framework.history.prev_size,
                                                                        next_t - curr_t, status=status))
        curr_t = next_t
        t += 1

    if eval or not (log_compact or log_coverage_set_only):
        agent_logger.write_data(log_entries, path)
    if discount_history_logger:
        discount_history_logger.write_data(history_entries)
    return transitions


def eval_(env, model, coverage_set, horizons, max_return, agent_logger, discount_history_logger, current_ep, current_t,
          gamma=1., n=10, normalise_state=False, eval_axes=False, log_compact=False, log_coverage_set_only=False):
    e_returns = np.empty((coverage_set.shape[0], n, coverage_set.shape[-1]))
    all_transitions = []
    for e_i, target_return, horizon in zip(np.arange(len(coverage_set)), coverage_set, horizons):
        n_transitions = []
        for n_i in range(n):
            transitions = run_episode_fairness(env, model, target_return, np.float32(horizon), max_return, agent_logger,
                                               discount_history_logger, current_ep, current_t,
                                               eval=True, normalise_state=normalise_state,
                                               eval_axes=eval_axes, log_compact=log_compact,
                                               log_coverage_set_only=log_coverage_set_only)
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
               use_wandb=True,
               log_compact=False,
               log_coverage_set_only=False,
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
    discount_history_logger = DiscountHistoryLogger() if env.fairness_framework.discount_factor else None
    env.fairness_framework.history.logger = discount_history_logger

    # agent_logger.create_file(agent_logger.path_eval_axes)
    if not log_coverage_set_only:
        agent_logger.create_file(agent_logger.path_eval)
    if not (log_compact or log_coverage_set_only):
        agent_logger.create_file(agent_logger.path_train)
        agent_logger.create_file(agent_logger.path_experience)

    # leaves_logger.create_file(f"{logdir}/leaves_log.csv")
    pcn_logger.create_file(f"{logdir}/pcn_log.csv")
    if not log_coverage_set_only:
        eval_logger.create_file(f"{logdir}/eval_log.csv")
    if discount_history_logger:
        discount_history_logger.create_file(f"{logdir}/history.csv")
    n_checkpoints = 0

    # fill buffer with random episodes
    experience_replay = []
    print("Experience replay...")

    for ep in range(n_er_episodes):
        curr_t = time.time()
        log_entries = []
        history_entries = []
        transitions = []
        obs = env.reset()
        done = False
        while not done:
            curr_obs = env.normalise_state(obs) if normalise_state else obs
            action = np.random.randint(0, env.nA)
            n_obs, reward, done, info = env.step(action, scores=np.full(env.nA, fill_value=1 / env.nA))
            next_obs = env.normalise_state(n_obs) if normalise_state else n_obs
            # TODO
            if step % 100 == 0:
                print("t=", step, ep, action, reward)

            transitions.append(
                Transition(curr_obs if normalise_state else curr_obs.to_array(), action, np.float32(reward).copy(),
                           next_obs if normalise_state else next_obs.to_array(), done))
            next_t = time.time()

            if not (log_compact or log_coverage_set_only):
                log_entries.append(agent_logger.create_entry(ep, step, obs, action, reward, done, info, next_t - curr_t,
                                                             status="e_replay"))
            if discount_history_logger:
                history_entries.append(discount_history_logger.create_entry(ep, step,
                                                                            env.fairness_framework.history.get_size(),
                                                                            env.fairness_framework.history.difference,
                                                                            env.fairness_framework.history.prev_size,
                                                                            next_t - curr_t, status="e_replay"))

            curr_t = next_t

            obs = n_obs
            step += 1
        # add episode in-place
        print(f"Store episode {ep}, t {step}")
        add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
        if not (log_compact or log_coverage_set_only):
            agent_logger.write_data(log_entries, agent_logger.path_experience)
        if discount_history_logger:
            discount_history_logger.write_data(history_entries)

    del log_entries

    # return  # TODO
    print("Training...")
    update_num = 0
    print_update_interval = 5

    while step < total_steps:
        if update_num % print_update_interval == 0:
            print("loop", update_num)

        loss = []
        entropy = []
        for moupd in range(n_model_updates):
            l, lp = update_model_hire(model, opt, experience_replay, batch_size=batch_size)
            loss.append(l)
            lp = lp
            ent = np.sum(-np.exp(lp) * lp)
            entropy.append(ent)

        desired_return, desired_horizon = choose_commands(experience_replay, n_er_episodes, objectives)

        e_lengths, e_returns = [(len(e[2])) for e in experience_replay[len(experience_replay) // 2:]], \
                               [(e[2][0].reward) for e in experience_replay[len(experience_replay) // 2:]]
        e_lengths, e_returns = np.array(e_lengths), np.array(e_returns)
        try:
            if len(experience_replay) == max_size:
                if use_wandb:
                    logger.put('train/leaves', e_returns, step, f'{e_returns.shape[-1]}d')
                else:
                    leaves = []
                    # get all leaves, contain biggest elements, experience_replay got heapified in choose_commands
                    # print([(len(e[2]), e[2][0].reward) for e in experience_replay[len(experience_replay) // 2:]])
                    # leaves = np.array([(len(e[2]), e[2][0].reward)
                    #                    for e in experience_replay[len(experience_replay) // 2:]])
                    # e_lengths, e_returns = zip(*leaves)
                    # print([(len(e[2]), e[2][0].reward) for e in experience_replay[len(experience_replay) // 2:]])
                    # leaves = np.array([(len(e[2]), e[2][0].reward)
                    #                    for e in experience_replay[len(experience_replay) // 2:]])
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
                                               discount_history_logger,
                                               normalise_state=normalise_state, current_t=step, current_ep=ep,
                                               log_compact=log_compact, log_coverage_set_only=log_coverage_set_only)
            step += len(transitions)
            ep += 1
            add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
            returns.append(transitions[0].reward)
            horizons.append(len(transitions))

        print(f'step {step}/{int(total_steps)} ({round((step + 1) / total_steps * 100, 3)}%) '
              f'\t return {np.mean(returns, axis=0)}, ({np.std(returns, axis=0)}) \t loss {np.mean(loss):.3E}')

        # compute hypervolume of leaves
        valid_e_returns = e_returns[np.all(e_returns[:, objectives] >= ref_point[objectives, ], axis=1)]
        hv = compute_hypervolume(np.expand_dims(valid_e_returns[:, objectives], 0), ref_point[objectives, ])[0] if len(
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
            e_r, t_r = eval_(env, model, e_returns, e_lengths, max_return, agent_logger, discount_history_logger, ep, step,
                             gamma=gamma, n=n_evaluations, normalise_state=normalise_state, log_compact=log_compact)

            # compute e-metric
            epsilon = epsilon_metric(e_r[..., objectives].mean(axis=1), e_returns[..., objectives])
            print('\n', '=' * 10, f' evaluation (t={step}) ', '=' * 10, sep='')
            for d, r in zip(e_returns, e_r):
                print('desired: ', d, '\t', 'return: ', r.mean(0))
            print(f'epsilon max/mean: {epsilon.max():.3f} \t {epsilon.mean():.3f}')
            print('=' * 22, '\n', sep='')

            if not (log_compact or log_coverage_set_only):
                entries = []
                for d, r in zip(e_returns, e_r):
                    entry = eval_logger.create_entry(ep, step, epsilon.max(), epsilon.mean(), d, r.mean(0), "eval")
                    entries.append(entry)
                eval_logger.write_data(entries)

        update_num += 1


if __name__ == '__main__':
    import time
    t_start = time.time()

    parser = argparse.ArgumentParser(description='PCN-Fair', formatter_class=argparse.RawDescriptionHelpFormatter,
                                     parents=[fMDP_parser])
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--steps', default=1e5, type=float, help='total timesteps')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--model-updates', default=20, type=int,
                        help='number of times the model is updated at every training iteration')
    parser.add_argument('--top-episodes', default=50, type=int,
                        help='top-n episodes used to compute target-return and horizon. '
                             'Initially fill ER with n random episodes')
    parser.add_argument('--n-episodes', default=10, type=int,
                        help='number of episodes to run between each training iteration')
    parser.add_argument('--er-size', default=100, type=int,
                        help='max size (in episodes) of the ER buffer')
    parser.add_argument('--threshold', default=0.02, type=float, help='crowding distance threshold before penalty')
    parser.add_argument('--noise', default=0.0, type=float, help='noise applied on target-return on batch-update')
    parser.add_argument('--model', default='densesmall', type=str, help='dense(big|small)')

    args = parser.parse_args()
    no_save = False
    args.wandb = 0

    # ########
    # args.vsc = 0
    #
    # args.steps = 5000
    # args.window = 500
    # args.team_size = 100
    # args.top_episodes = 10
    # args.n_episodes = 15
    # args.er_size = 30
    # args.episode_length = args.team_size * 10
    # args.env = "fraud"
    # args.n_transactions = 200
    # args.fraud_proportion = 0.20
    #
    # args.top_episodes = 15
    # args.n_episodes = 15
    # args.er_size = 200
    # args.model_updates = 10
    #
    # args.objectives = ["R", "SP", "IF"]#, "IF", "IF"]
    # args.compute_objectives = ["EO"]
    # args.distance_metrics = ["HMOM"] * 2
    # args.distance_metrics = ["braycurtis", "HMOM"]#, "HEOM"]
    # args.steps = 5000
    # args.window = 1000
    # args.bias = 1
    # args.ignore_sensitive = True
    # args.log_compact = True
    # args.compute_individual = True
    # args.combined_sensitive_attributes = 0
    # args.log_dir = f"knn_graph{args.combined_sensitive_attributes}"
    # args.log_dir = f"discount_debug"
    # args.log_coverage_set_only = True
    # args.discount_history = True
    # args.discount_factor = 0.95
    # args.discount_threshold = 1e-5

    print(args)

    arg_use_wandb = args.wandb == 1
    on_vsc = args.vsc == 1

    env_type = args.env
    is_job_hiring = env_type == "job"
    n_evaluations = 10

    env, logdir, ref_point, scaling_factor, max_return = create_fairness_framework_env(args)
    print(args)

    kw = "small" if args.model == "densesmall" else "big"
    ss, se, sa = ss_emb[env_type][kw], se_emb[kw], sa_emb[kw]

    model = Model(env.nA, scaling_factor, tuple(args.objectives), ss).to(device)
    model = DiscreteHead(model)

    # from cProfile import Profile
    # with Profile() as pr:
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
               use_wandb=arg_use_wandb,
               log_compact=args.log_compact,
               log_coverage_set_only=args.log_coverage_set_only,
               )
    # pr.print_stats(sort="cumulative")

    t_end = time.time()
    print(t_end - t_start, "seconds")
