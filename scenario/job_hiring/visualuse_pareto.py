import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_rows_required(elements, columns):
    return int(np.ceil(elements / columns))


def get_xy_index(idx, rows, columns, fill_row_first=True):
    if fill_row_first:
        idx_row = idx // columns
        idx_col = idx % columns
    else:
        idx_row = idx // rows
        idx_col = idx % rows
    return idx_row, idx_col


def plot_radar_pf(df, n_cols=4):
    """Produce radar plots of all trade-offs of a learned pareto front"""
    # n_cols = 4
    n_rows = get_rows_required(len(df), n_cols)
    titles = [f"Returns {n}" for n in range(len(df))]
    min_o = df.to_numpy().min()
    max_o = df.to_numpy().max()

    fig = make_subplots(rows=n_rows, cols=n_cols, vertical_spacing=0.2 / n_rows, subplot_titles=titles,
                        specs=[[{"type": "polar"} for _ in range(n_cols)] for _ in range(n_rows)],
                        # shared_xaxes=True, shared_yaxes=True,
                        )
    full_fig = go.Figure()

    # Plot data
    for idx, row in df.iterrows():
        idx_row, idx_col = get_xy_index(idx, n_rows, n_cols)
        fig.add_trace(go.Scatterpolar(r=row.values, theta=row.keys(), fill='toself', name=f"Return {idx}", showlegend=False),
                      row=idx_row + 1, col=idx_col + 1)
        full_fig.add_trace(go.Scatterpolar(r=row.values, theta=row.keys(), fill='toself', name=f"Return {idx}"))

        fig.update_layout({
            f"polar{idx + 1 if idx != 0 else ''}": dict(radialaxis=dict(visible=True, range=[min_o, max_o])),

        })

    fig.update_layout(height=300 * n_rows)

    return full_fig, fig



if __name__ == '__main__':
    import pandas as pd
    import dash
    from dash import dcc, html
    import plotly.graph_objects as go
    import plotly.express as px
    import pickle

    is_biased = False
    # is_biased = True

    if is_biased:
        file = "./wandb/run-20230413_205514-2rejonh4/files/media/table/nd_coverage_set_10230_6eae8cf1df45730deefa.table.json"
        res_file = "./pcn/tmp/jobhiring/objectives_[0, 1, 2, 8]/env_ode/action_discrete/lr_0.01/steps_10000.0/batch_128/model_updates_20/top_episodes_10/n_episodes_10/er_size_100/threshold_0.02/noise_0.0/model_conv1dsmall/2023-04-13_20-55-12_df06/fair_results_10.pt"
    else:
        file = "./wandb/run-20230413_203734-0fxery45/files/media/table/nd_coverage_set_10297_4a7aa4c65b255ab2d053.table.json"
        res_file = "./pcn/tmp/jobhiring/objectives_[0, 1, 2, 8]/env_ode/action_discrete/lr_0.01/steps_10000.0/batch_128/model_updates_20/top_episodes_10/n_episodes_10/er_size_100/threshold_0.02/noise_0.0/model_conv1dsmall/2023-04-13_20-37-32_4634/fair_results_10.pt"

    file = "./wandb/run-20230417_121755-zr0n4c5c/files/media/table/nd_coverage_set_10135_a0c77e59443e3c01f4ca.table.json"
    res_file = "./pcn/tmp/jobhiring/objectives_[0, 1]/env_ode/action_discrete/lr_0.01/steps_10000.0/batch_128/model_updates_20/top_episodes_10/n_episodes_10/er_size_100/threshold_0.02/noise_0.0/model_conv1dsmall/2023-04-17_12-17-53_6cf6/fair_results_10.pt"
    # file = "./nd_coverage_set.json"
    # file = "./executions.json"
    df = pd.read_json(file, orient="split")
    # df = df.rename(columns={'o_0': 'Reward', 'o_1': 'StatisticalParity', "o_2": "EqualOpportunity"})
    # TODO: reward is scaled, but scale back down within fairness notions for the radarplot
    # df["reward"] /= 5
    # print(df)

    # res_file = "./pcn/tmp/jobhiring/objectives_[0, 1, 2, 8]/env_ode/action_discrete/lr_0.01/steps_10000.0/batch_128/model_updates_20/top_episodes_10/n_episodes_10/er_size_100/threshold_0.02/noise_0.0/model_conv1dsmall/2023-04-13_20-14-35_93e2/" \
    #            "fair_results_1.pt"
    with open(res_file, "rb") as f:
        results = pickle.load(f)
    # print(results["main_reward"])

    ts = []
    cr = []
    episodes = results["main_reward"]
    eps_dfs = [pd.DataFrame(r, columns=df.columns) for r in episodes]
    for ep, edf in enumerate(eps_dfs):
        edf["episode"] = [ep] * len(edf)
        edf["t"] = [_ for _ in range(len(edf))]
        ts.append(len(edf))
        edf["cr"] = edf["reward"].cumsum()
        cr.append(edf["reward"].sum())
    full_df = pd.concat(eps_dfs)
    # print(eps_dfs[0])
    # TODO: fix episode counter
    # r = 1000
    # full_df["t"] = [n for n in range(len(full_df))]
    # full_df["range"] = full_df["t"] // r
    # f = (len(full_df) // r)
    # rem = len(full_df) - (f * r)
    # full_df["t"] = [n for n in range(r)] * f + [n for n in range(rem)]

    # TODO: reward is scaled, but scale back down within fairness notions for the radarplot
    # full_df["reward"] /= 5

    # full_fig = px.line(full_df, x="t", y=df.columns,
    #                       animation_frame="range", # animation_frame="episode",
    #                       # animation_group="country",
    #                       # hover_name="country",
    #                       # range_x=[0, 100],
    #                     # range_y=[-1, 1]
    #                       )

    # full_fig = px.line(full_df, x="t", y=df.columns,
    #                       animation_frame="episode",
    #                       # animation_group="country",
    #                       # hover_name="country",
    #                       range_x=[0, 100],
    #                     # range_y=[-1, 1]
    #                       )

    full_fig = px.line(y=cr)


    # radar_plots = []
    # for idx, row in df.iterrows():
    #     fig = px.line_polar(r=row.values, theta=row.keys(), line_close=True)
    #     fig.update_traces(fill='toself')
    #     fig = dcc.Graph(figure=fig)
    #     radar_plots.append(fig)

    radar_fig_full, radar_figs = plot_radar_pf(df, n_cols=3)

    # hist_fig = px.histogram(full_df, x="reward")
    hist_fig = px.histogram(ts)

    cumul_fig = px.line(full_df, x="t", y=["cr"],
                          animation_frame="episode",
                          # animation_group="country",
                          # hover_name="country",
                          range_x=[0, 100],
                        # range_y=[-1, 1]
                        )

    app = dash.Dash()
    app.layout = html.Div([
        # dcc.Graph(figure=results["overview"]),
        dcc.Graph(figure=hist_fig),
        dcc.Graph(figure=cumul_fig),
        dcc.Graph(figure=full_fig),
        # *cms,
        # *radar_plots,
        dcc.Graph(figure=radar_fig_full),
        dcc.Graph(figure=radar_figs),
    ])

    app.run_server(debug=False, use_reloader=False)

