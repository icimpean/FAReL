import sys
sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from agent.dqn import Experience
from agent.pcn.pcn_core import epsilon_metric, non_dominated, compute_hypervolume, add_episode, \
    choose_commands, Transition
from agent.pcn.logger import Logger
from create_fair_env import *
from fairness.fairness_framework import ExtendedfMDP
from loggers.logger import AgentLogger, LeavesLogger, TrainingPCNLogger, EvalLogger, DiscountHistoryLogger
import pcn_setup as pcn
import dqn_setup as dqn


def get_model_and_methods(args, env_type, n_actions, scaling_factor):
    if args.name == "PCN":
        model = pcn.get_model(args, env_type, n_actions, scaling_factor)
        choose_action = pcn.choose_action
        update_model = pcn.update_model
    elif args.name == "DQN":
        model = dqn.get_model(args, env_type, n_actions)
        choose_action = dqn.choose_action
        update_model = dqn.update_model
    else:
        raise ValueError(f"Expected 'PCN' or 'DQN', given: {args.name}")
    print(f"Initialised {args.name} model...")

    return model, choose_action, update_model


def run_episode(env, model, is_pcn, choose_action, desired_return, desired_horizon, max_return,
                agent_logger, discount_history_logger, current_ep, current_t, eval=False, normalise_state=False,
                eval_axes=False, log_compact=False, log_coverage_set_only=False):
    curr_t = time.time()
    transitions = []
    experiences = 0
    ep_returns = np.zeros_like(max_return)
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
        if is_pcn:
            action, scores = choose_action(model, curr_obs if normalise_state else curr_obs.to_array(),
                                           desired_return, desired_horizon, eval=eval, return_probs=True)
        else:
            action, scores = choose_action(model, curr_obs if normalise_state else curr_obs.to_array(),
                                           current_t, current_ep, eval=eval, return_probs=True)
        n_obs, reward, done, info = env.step(action, scores)
        next_obs = env.normalise_state(n_obs) if normalise_state else n_obs

        transitions.append(Transition(
            observation=curr_obs if normalise_state else curr_obs.to_array(),
            action=action,
            reward=np.float32(reward).copy(),
            next_observation=next_obs if normalise_state else next_obs.to_array(),
            terminal=done
        ))
        if not is_pcn:
            model.store_experience(Experience(curr_obs if normalise_state else curr_obs.to_array(), action,
                                              np.float32(reward[0]).copy(),  # TODO
                                              done, next_obs if normalise_state else next_obs.to_array(),
                                              model.predict_q_values(
                                                  curr_obs if normalise_state else curr_obs.to_array())))
            experiences += 1
            ep_returns += reward

        obs = n_obs
        if is_pcn:
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
    return transitions if is_pcn else (transitions, experiences, ep_returns)


def eval_(env, model, is_pcn, choose_action, coverage_set, horizons, max_return, agent_logger, discount_history_logger,
          current_ep, current_t,
          gamma=1., n=10, normalise_state=False, eval_axes=False, log_compact=False, log_coverage_set_only=False):
    e_returns = np.empty((coverage_set.shape[0], n, coverage_set.shape[-1]))
    all_transitions = []
    for e_i, target_return, horizon in zip(np.arange(len(coverage_set)), coverage_set, horizons):
        n_transitions = []
        for n_i in range(n):
            transitions = run_episode(env, model, is_pcn, choose_action, target_return, np.float32(horizon), max_return,
                                      agent_logger, discount_history_logger, current_ep, current_t,
                                      eval=True, normalise_state=normalise_state, eval_axes=eval_axes,
                                      log_compact=log_compact, log_coverage_set_only=log_coverage_set_only)
            if not is_pcn:
                transitions = transitions[0]
            # compute return
            for i in reversed(range(len(transitions) - 1)):
                transitions[i].reward += gamma * transitions[i + 1].reward
            e_returns[e_i, n_i] = transitions[0].reward
            n_transitions.append(transitions)
        all_transitions.append(n_transitions)

    return e_returns, all_transitions


def train_fair(env, name, model, choose_action, update_model, learning_rate=1e-2, batch_size=1024, total_steps=1e4,
               n_model_updates=100, n_step_episodes=10, n_er_episodes=10,
               gamma=1., max_return=250., max_size=500, ref_point=np.array([0, 0]),
               threshold=0.2, noise=0.0,
               objectives=None, n_evaluations=10, logdir='runs/', normalise_state=False,
               use_wandb=True, log_compact=False, log_coverage_set_only=False):
    step = 0
    is_pcn = name == "PCN"
    if objectives == None:
        objectives = tuple([i for i in range(len(ref_point))])
    total_episodes = n_er_episodes
    if is_pcn:
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
    pcn_logger.create_file(f"{logdir}/{name.lower()}_log.csv")
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
            if not is_pcn:
                # Experience = namedtuple("Experience", "state, action, reward, done, next_state, q_values")
                model.store_experience(Experience(curr_obs if normalise_state else curr_obs.to_array(), action,
                                                  np.float32(reward[0]).copy(), done,  # TODO
                                                  next_obs if normalise_state else next_obs.to_array(),
                                                  model.predict_q_values(
                                                      curr_obs if normalise_state else curr_obs.to_array())))
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

        if not is_pcn:
            model.set_training(True)

        loss = []
        entropy = []
        for moupd in range(n_model_updates):
            if is_pcn:
                l, lp = update_model(model, opt, experience_replay, batch_size=batch_size)
                ent = np.sum(-np.exp(lp) * lp)
            else:
                l = model.train(step)
                ent = 0
            loss.append(l)
            entropy.append(ent)

        e_lengths, e_returns = [(len(e[2])) for e in experience_replay[len(experience_replay) // 2:]], \
                               [(e[2][0].reward) for e in experience_replay[len(experience_replay) // 2:]]
        e_lengths, e_returns = np.array(e_lengths), np.array(e_returns)
        desired_return, desired_horizon = choose_commands(experience_replay, n_er_episodes, objectives)
        if is_pcn:
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
                transitions = run_episode(env, model, is_pcn, choose_action, desired_return, desired_horizon,
                                          max_return,
                                          agent_logger, discount_history_logger,
                                          normalise_state=normalise_state, current_t=step, current_ep=ep,
                                          log_compact=log_compact, log_coverage_set_only=log_coverage_set_only)
                step += len(transitions)
                ep += 1
                add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
                returns.append(transitions[0].reward)
                horizons.append(len(transitions))
        else:
            returns = []
            horizons = []
            for _ in range(n_step_episodes):
                # desired_return, desired_horizon = [0], [0]
                transitions, experiences, ep_return = run_episode(env, model, is_pcn, choose_action,
                                                                  desired_return, desired_horizon,
                                                                  max_return, agent_logger, discount_history_logger,
                                                                  normalise_state=normalise_state, current_t=step,
                                                                  current_ep=ep, log_compact=log_compact,
                                                                  log_coverage_set_only=log_coverage_set_only)
                step += experiences
                ep += 1
                add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
                returns.append(ep_return)
                horizons.append(experiences)

        print(f'step {step}/{int(total_steps)} ({round((step + 1) / total_steps * 100, 3)}%) '
              f'\t return {np.mean(returns, axis=0)}, ({np.std(returns, axis=0)}) \t loss {np.mean(loss):.3E}')

        # if is_pcn:
        # compute hypervolume of leaves
        valid_e_returns = e_returns[np.all(e_returns[:, objectives] >= ref_point[objectives,], axis=1)]
        hv = compute_hypervolume(np.expand_dims(valid_e_returns[:, objectives], 0), ref_point[objectives,])[
            0] if len(
            valid_e_returns) and len(objectives) > 1 else 0
        # else:
        #     hv = None

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

            if not is_pcn:
                model.set_training(False)

            columns = env.obj_names if isinstance(env, ExtendedfMDP) else [f'o_{o}' for o in range(e_returns.shape[1])]

            # # current coverage set
            # _, e_i = non_dominated(e_returns[:, objectives], return_indexes=True)
            e_returns = e_returns[e_i]
            e_lengths = e_lengths[e_i]
            e_r, t_r = eval_(env, model, is_pcn, choose_action, e_returns, e_lengths, max_return,
                             agent_logger, discount_history_logger, ep, step,
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

    parser = argparse.ArgumentParser(description='FairRL', formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.add_argument('--name', default='PCN', type=str, help='The name of the model to use (PCN or DQN).')

    args = parser.parse_args()
    no_save = False
    args.wandb = 0

    # ########
    # args.vsc = 0
    #
    # args.steps = 2000
    # args.window = 500
    # args.team_size = 100
    # args.top_episodes = 5
    # args.n_episodes = 10
    # args.er_size = 15
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
    # args.name = "DQN"
    # args.objectives = ["R"]  # , "IF", "IF"]
    # args.compute_objectives = ["SP:EO:OAE:PP:IF:CSC"]
    # args.distance_metrics = ["HEOM"] * 2
    # args.steps = 15000
    # args.window = 1000
    # args.distance_metrics = ["braycurtis", "HMOM"]#, "HEOM"]
    # args.steps = 5000
    # args.window = 1000
    # args.bias = 1
    # args.ignore_sensitive = True
    # args.log_compact = True
    # args.compute_individual = True
    # args.combined_sensitive_attributes = 0
    # args.log_dir = f"DQN_single"
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

    model, choose_action, update_model = get_model_and_methods(args, env_type, env.nA, scaling_factor)

    # from cProfile import Profile
    # with Profile() as pr:
    train_fair(env, args.name,
               model, choose_action, update_model,
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
               log_coverage_set_only=args.log_coverage_set_only)
    # pr.print_stats(sort="cumulative")

    t_end = time.time()
    print(t_end - t_start, "seconds")

#          22646947 function calls (22458600 primitive calls) in 27.400 seconds
#
#    Ordered by: cumulative time
#
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         1    0.013    0.013   27.585   27.585 main_pcn_core.py:289(train_fair)
#       155    0.118    0.001   25.540    0.165 main_pcn_core.py:121(run_episode_fairness)
#      5272    0.089    0.000   24.390    0.005 fairness_framework.py:156(step)
#         3    0.012    0.004   17.348    5.783 main_pcn_core.py:185(eval_)
#     10544    0.014    0.000   15.144    0.001 fairness_framework.py:109(get_individual_notion)
#     10544    0.051    0.000   15.130    0.001 individual_fairness.py:117(get_notion)
#      5272    1.243    0.000   13.897    0.003 individual_fairness.py:123(individual_fairness)
#      5272    0.140    0.000   10.292    0.002 individual_fairness.py:164(<listcomp>)
#    516978    0.438    0.000   10.152    0.000 individual_fairness.py:48(_pool_individual_fairness)
#      5272    0.064    0.000    6.791    0.001 env.py:156(step)
#    516978    0.173    0.000    6.019    0.000 __init__.py:273(similarity_metric)
#    516978    0.366    0.000    5.846    0.000 fairness_framework.py:129(<lambda>)
#    516978    4.221    0.000    5.479    0.000 __init__.py:290(H_OM_distance)
#    516978    3.695    0.000    3.695    0.000 individual_fairness.py:35(hellinger)
#      5437    0.044    0.000    3.636    0.001 env.py:230(generate_sample)
#      5437    0.059    0.000    3.564    0.001 features.py:174(sample)
#      8737    0.301    0.000    2.964    0.000 env.py:270(generate_company_state)
#     21088    0.021    0.000    2.034    0.000 fairness_framework.py:104(get_group_notion)
#     21088    0.024    0.000    2.012    0.000 group_fairness.py:45(get_notion)
#      5437    0.059    0.000    1.987    0.000 env.py:340(calc_goodness)
#     21088    0.135    0.000    1.954    0.000 __init__.py:79(_fairness_notion)
#   1033956    1.428    0.000    1.738    0.000 individual_deque.py:25(append)
#      4891    0.153    0.000    1.533    0.000 main_pcn_core.py:250(choose_action_hire)
#      5437    0.017    0.000    1.417    0.000 features.py:168(_sample)
#     26211    0.490    0.000    1.415    0.000 env.py:237(_entropy)
#      5272    0.066    0.000    1.175    0.000 individual_fairness.py:359(consistency_score_complement)
# 74265/4951    0.110    0.000    1.042    0.000 module.py:1494(_call_impl)
#      4951    0.018    0.000    1.019    0.000 main_pcn_core.py:115(forward)
#     17338    0.681    0.000    0.987    0.000 {method 'choice' of 'numpy.random._generator.Generator' objects}
#      5437    0.016    0.000    0.986    0.000 indexing.py:1089(__getitem__)
#     21088    0.152    0.000    0.970    0.000 history.py:198(get_confusion_matrices)
#      5437    0.016    0.000    0.958    0.000 indexing.py:1623(_getitem_axis)
#      4951    0.205    0.000    0.941    0.000 main_pcn_core.py:96(forward)
#     10874    0.036    0.000    0.915    0.000 frame.py:1354(iterrows)
#      5437    0.006    0.000    0.875    0.000 indexing.py:1600(_get_list_axis)
#      5437    0.015    0.000    0.869    0.000 generic.py:3940(_take_with_is_copy)
#     26053    0.040    0.000    0.826    0.000 arraysetops.py:138(unique)
#      5437    0.031    0.000    0.746    0.000 generic.py:3911(_take)
#    580274    0.726    0.000    0.726    0.000 {built-in method builtins.sum}
