import random

import numpy as np
import pandas as pd
import torch

from agent.dqn import DQNAgent, EpsilonGreedyPolicy, DecayingEpsilonGreedyPolicy
from agent.dqn.policy import BoltzmannPolicy
from fairness import SensitiveAttribute
from fairness.fairness_framework import FairnessFramework
from fairness.group import GroupNotion, ALL_GROUP_NOTIONS
from fairness.individual import IndividualNotion
from run_scenario import run_scenario
from scenario import FeatureBias
from scenario.job_hiring import run_scenario_dash
from scenario.job_hiring.features import HiringFeature, Gender, EMPLOYMENT_TRANSITIONS, EMPLOYMENT_DURATIONS
from scenario.job_hiring.env import JobHiringEnv
import dash
from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px

from scenario.job_hiring.visualuse_pareto import plot_radar_pf

if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Training environment
    # Baseline Hiring Scenario
    team_size = 20 * 2
    episode_length = 100 * 2
    diversity_weight = 0

    # team_size = None
    env_init = JobHiringEnv(team_size=team_size, episode_length=episode_length, diversity_weight=diversity_weight,
                            seed=seed)
    input_shape = env_init.input_shape
    actions = env_init.actions

    # TODO: test bias
    # env_init.reward_biases.append(FeatureBias(HiringFeature.gender, Gender.male, bias=0.1))
    # env_init.reward_biases.append(FeatureBias([HiringFeature.gender, HiringFeature.age],
    #                                           [Gender.male, lambda age: age > 25], bias=0.1))

    # TODO: leaving employees
    env_init.employment_transitions = EMPLOYMENT_TRANSITIONS

    env_train = JobHiringEnv(team_size=team_size, episode_length=episode_length, diversity_weight=diversity_weight,
                             seed=seed+1)
    env_train.employment_transitions = EMPLOYMENT_TRANSITIONS
    # Testing environment
    env_eval = env_train

    # steps_init = 100
    # steps_train = 400
    # steps_eval = 0

    steps_init = 500
    steps_train = 1000
    steps_eval = 500

    # steps_init = steps_train = steps_eval = 10000

    # policy = DecayingEpsilonGreedyPolicy(epsilon_max=1.0, epsilon_min=0.1, warm_up_steps=0, steps=steps_init + steps_train)
    # policy = EpsilonGreedyPolicy(epsilon=0.1)
    policy = BoltzmannPolicy(tau=0.1)

    agent = DQNAgent(input_shape, 64, actions, warm_up=100, policy=policy,
                     memory_size=steps_train, batch_size=64, learning_rate=0.005, seed=seed)

    # Gender
    # sensitive_feature = FraudFeature.gender
    # sensitive_value = Gender.female
    # other_value = Gender.male

    sensitive_attribute = SensitiveAttribute(HiringFeature.gender, sensitive_values=Gender.female,
                                             other_values=Gender.male)

    # No experience
    # sensitive_feature = HiringFeature.experience
    # sensitive_value = 0
    # other_value = set([exp for vals in test_env.experience_desc.ranges.values() for exp in vals if exp != sensitive_value])

    thr = 0.05
    alpha = 0.5

    fairness_frameworks = [
        None for _ in range(3)]
    # fairness_framework_w = [
    #     None for _ in range(3)]
    fairness_framework_w = [
        FairnessFramework(actions, sensitive_attribute,
                          group_notions=[n for n in ALL_GROUP_NOTIONS if n != GroupNotion.TreatmentEquality],
                          threshold=thr,
                          window=100,
                          # individual_notions=[],
                          individual_notions=[IndividualNotion.IndividualFairness],
                          # individual_notions=[IndividualNotion.IndividualFairness, IndividualNotion.ConsistencyScoreComplement],
                          # individual_notions=[IndividualNotion.IndividualFairness, IndividualNotion.WeaklyMeritocratic],
                          get_individual=env_init.get_individual,
                          similarity_metric=env_init.similarity_metric,
                          alpha=alpha,
                          visualise=True,
                          visualise_cm=True,
                          visualise_notions=True,
                          visualise_reward=True,
                          # visualise_hist=True
                          ) for _ in range(3)]

    # run_scenario(agent, env_init, env_train, env_eval, steps_init, steps_train, steps_eval,
    #              fairness_frameworks, fairness_framework_w)

    # steps_init = 100
    # steps_train = 100
    # steps_eval = 0

    import pickle
    f = "./results_baseline_leave_test.pickle"

    results_init, results_train, results_test = run_scenario_dash(agent, env_init, env_train, env_eval, steps_init, steps_train, steps_eval,
                                fairness_frameworks, fairness_framework_w)
    with open(f, mode="wb") as file:
        pickle.dump((results_init, results_train, results_test), file)

    # with open(f, mode="rb") as file:
    #     results_init, results_train, results_test = pickle.load(file)

    main_rewards = results_init["main_reward"]

    for res in (results_init, results_train, results_test):

        mrs = np.array([r for mr in res["main_reward"] for r in mr])
        for ep, rewards in enumerate(res["main_reward"]):
            rs = np.array(rewards)
            if len(rs) > 0:
                print(f"Ep {ep}: min {np.min(rs)} - max {np.max(rs)} - mean {np.mean(rs)}")
        print(f"Total: min {np.min(mrs)} - max {np.max(mrs)} - mean {np.mean(mrs)}")

    # graphs = [dcc.Graph(figure=fig) for fig in results_init]

    results = results_test

    cms = [dcc.Graph(figure=fig) for fig in results["cm"]]

    reward_fig = go.Figure()
    for fig, notion in zip(results["reward"], results["notions"]):
        t = list(fig.select_traces(lambda trace: trace.name != "threshold"))[0]
        t["line"]["color"] = None
        t["name"] = notion.name
        reward_fig.add_trace(t)
    reward_fig.add_scatter(x=t["x"], y=[-thr] * len(t["x"]), mode='lines', line_color='black', name=f'threshold',
                            showlegend=True)

    # file = "./objectives_test.csv"
    # df = pd.read_csv(file)
    # df = df.rename(columns={'o_0': 'Reward', 'o_1': 'StatisticalParity', "o_2": "EqualOpportunity"})
    # print(df)

    # radar_plots = []
    # for idx, row in df.iterrows():
    #     fig = px.line_polar(r=row.values, theta=row.keys(), line_close=True)
    #     fig.update_traces(fill='toself')
    #     fig = dcc.Graph(figure=fig)
    #     radar_plots.append(fig)

    # radar_fig_full, radar_figs = plot_radar_pf(df)

    # episodes = results["main_reward"]
    # eps_dfs = [pd.DataFrame(pd.Series(r, name="reward")) for r in episodes]

    eps_dfs = []
    for eps in (results_init, results_train, results_test):
        mr = eps["main_reward"]
        episodes = [pd.DataFrame(pd.Series(r, name="reward")) for r in mr]
        eps_dfs.extend(episodes)

    ts = []

    for ep, edf in enumerate(eps_dfs):
        edf["episode"] = [ep] * len(edf)
        edf["t"] = [_ for _ in range(len(edf))]

        ts.append(len(edf))

    full_df = pd.concat(eps_dfs)

    hist_fig = px.histogram(full_df, x="reward")
    # hist_fig = px.histogram(ts)

    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=hist_fig),
        dcc.Graph(figure=results["overview"]),
        dcc.Graph(figure=reward_fig),
        *cms,
        # *radar_plots,
        # dcc.Graph(figure=radar_fig_full),
        # dcc.Graph(figure=radar_figs),
    ])

    app.run_server(debug=False, use_reloader=False)

    # https://plotly.com/python/
