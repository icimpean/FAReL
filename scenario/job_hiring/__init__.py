from agent import Agent
from agent.dqn import Experience
from fairness import ConfusionMatrix
from fairness.dash_visualisation import FairnessVisualisationDash
from fairness.fairness_framework import FairnessFramework


def run_scenario_dash(agent: Agent, env_init, env_train, env_eval, steps_init, steps_train, steps_eval,
                      fairness_framework=None, fairness_framework_windowed=None):
    current_t = 0
    results_init = results_train = results_test = None
    if fairness_framework is None:
        fairness_framework = [None, None, None]
    if fairness_framework_windowed is None:
        fairness_framework_windowed = [None, None, None]

    if steps_init > 0:
        print("Initialising...")
        # agent.training = True
        agent.set_training(train=True)
        current_t, results_init = _run(agent, current_t, steps_init, env_init, fairness_framework[0],
                                       fairness_framework_windowed[0])

    if steps_train > 0:
        print("TRAINING...")
        # agent.training = True
        agent.set_training(train=True)
        current_t, results_train = _run(agent, current_t, steps_train, env_train, fairness_framework[1],
                                        fairness_framework_windowed[1])

    if steps_eval > 0:
        print("Evaluating...")
        # agent.training = False
        agent.set_training(train=False)
        current_t, results_test = _run(agent, current_t, steps_eval, env_eval, fairness_framework[2],
                                       fairness_framework_windowed[2])

    return results_init, results_train, results_test


def _run(agent, current_t, steps, env, fairness_framework: FairnessFramework,
         fairness_framework_windowed: FairnessFramework):
    results = {
        'notions': [],
        'description': [],
        'reward': [],
        'cm': [],
        'overview': None,
        'main_reward': [[]],
    }

    timesteps = []  # TODO

    state = env.reset()
    episode = 0
    for t in range(current_t, current_t + steps):
        # print(state)

        # Select an action
        action, q_values = agent.select_action(state, t, episode=episode, return_q_values=True)
        scores = agent.policy.get_distribution([q_values], t, episode=episode)[0]
        # Observe the new state and reward
        next_state, reward, done, info = env.step(action)

        # Store the experience in memory and train the agent
        experience = Experience(state, action, reward, done, next_state, q_values)
        agent.store_experience(experience)
        agent_loss = agent.train(t)

        # Fairness framework
        if fairness_framework is not None:
            # print("fairness_framework update", t)
            fairness_framework.update_history(state, action, info["true_action"], scores, reward)
        if fairness_framework_windowed is not None:
            # print("fairness_framework_windowed update", t)
            fairness_framework_windowed.update_history(state, action, info["true_action"], scores, reward)

        notion_info = ""
        # for notion in notions:
        #     (exact, approx), diff, (prob_sensitive, prob_other) = \
        #         fairness_framework_windowed.get_group_notion(notion, sensitive_feature, sensitive_value, other_value,
        #                                                      thr)
        #     # notion_reward = -diff
        #     # notion_info += f", {notion.name}={notion_reward:3f} {'(E)' if exact else ('(F)' if approx else '')}"

        ep_info = f"ep {episode} | " if env.episode_length is not None else ""
        tab = '\n\t'
        # print(f"t={t:5d} | action={env.actions[action].name} ({action}), reward={reward:5f}"
        #       f"{tab if notion_info != '' else ''}{notion_info}")
        print(
            f"t={t:5d} | {ep_info}action={env.actions[action].name} ({action}), reward={reward:5f} (true_action: {info['true_action'], info['goodness']})"
            f"{tab if notion_info != '' else ''}{notion_info}")
        state = next_state
        results["main_reward"][-1].append(reward)

        timesteps.append([t, episode, str(state), env.actions[action].name, action, reward,
                          info['true_action'], info['goodness']])

        #
        if done:
            state = env.reset()
            episode += 1
            results["main_reward"].append([])

    import csv  # TODO
    with open(f"rl_interactions_t{current_t}.csv", mode="w") as file:
        csv_reader = csv.writer(file)
        csv_reader.writerow(["t", "episode", "state", "action name", "action", "reward", 'true_action', 'goodness'])
        csv_reader.writerows(timesteps)

    vis = FairnessVisualisationDash(env.actions)
    CM = ConfusionMatrix(env.actions)

    # Both the full + windowed fairness frameworks
    all_frameworks = [fairness_framework, fairness_framework_windowed]
    for framework in all_frameworks:
        if framework:
            h_st, h_a, h_ta, h_sc, h_r = framework.history.get_history()
            samples_X, samples_y_pred, samples_y_true = list(h_st), h_a, h_ta

            if framework.visualise_notions:
                # Individual fairness
                for notion in framework.individual_notions:
                    (exact, approx), diff, (u_ind, u_pairs, U_diff) = \
                        framework.get_individual_notion(notion, framework.get_individual, framework.threshold,
                                                        framework.similarity_metric, framework.alpha)
                    n = vis.print_individual_notion(notion, framework.threshold, exact, approx, diff, u_ind, u_pairs)
                    results["notions"].append(notion)
                    results["description"].append(n)
                    if framework.visualise_reward:
                        n = vis.fairness_notion_reward(framework, notion, framework.threshold, sensitive_attribute=None)
                        results["reward"].append(n)

            # For each sensitive attribute
            for sensitive_attribute in framework.sensitive_attributes:
                sensitive_feature = sensitive_attribute.feature
                sensitive_value = sensitive_attribute.sensitive_values
                other_value = sensitive_attribute.other_values

                # Full confusion matrix
                cm = CM.confusion_matrix(samples_X, samples_y_pred, samples_y_true)
                cm = vis.confusion_matrix(cm, sensitive_feature=None, sensitive_value=sensitive_value,
                                          other_value=other_value,
                                          is_sensitive_value=True, print_cm=True, plot_cm=framework.visualise_cm)
                results["cm"].append(cm)

                # Sensitive attribute
                cm = CM.confusion_matrix(samples_X, samples_y_pred, samples_y_true, sensitive_feature, sensitive_value)
                cm = vis.confusion_matrix(cm, sensitive_feature=sensitive_feature, sensitive_value=sensitive_value,
                                          other_value=other_value, is_sensitive_value=True, print_cm=True,
                                          plot_cm=framework.visualise_cm)
                results["cm"].append(cm)
                # Others
                if other_value is None:
                    cm = CM.confusion_matrix(samples_X, samples_y_pred, samples_y_true, sensitive_feature,
                                             sensitive_value, excluded=True)
                else:
                    cm = CM.confusion_matrix(samples_X, samples_y_pred, samples_y_true, sensitive_feature, other_value)
                cm = vis.confusion_matrix(cm, sensitive_feature=sensitive_feature, sensitive_value=sensitive_value,
                                          other_value=other_value, is_sensitive_value=False, print_cm=True,
                                          plot_cm=framework.visualise_cm)
                results["cm"].append(cm)

                if framework.visualise_notions:
                    # For each group notion for each attribute
                    for notion in framework.group_notions:
                        (exact, approx), diff, (prob_sensitive, prob_other) = \
                            framework.get_group_notion(notion, sensitive_attribute, framework.threshold)
                        n = vis.print_group_notion(notion, sensitive_feature, sensitive_value, other_value,
                                                   framework.threshold, exact, approx,
                                                   prob_sensitive, prob_other)
                        results["notions"].append(notion)
                        results["description"].append(n)

                        if framework.visualise_reward:
                            n = vis.fairness_notion_reward(framework, notion, framework.threshold, sensitive_attribute)
                            results["reward"].append(n)

            if framework.visualise:
                fig_radar = vis.get_fairness_radar_plot(framework, framework.all_notions, framework.threshold,
                                                        framework.sensitive_attributes)
                results["overview"] = fig_radar

            if framework.visualise_hist:
                for f in framework.history.states[0].get_state_features(no_hist=True):
                    vis.get_histogram_plot(framework.history, f, None, None)

    return t + 1, results
