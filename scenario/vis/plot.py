import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scenario.vis import find_representative_subset, reward_type, group_type, ind_type, TYPE_NOTION, type_columns


def plot_radar(requested_objectives, all_objectives, sorted_objectives, iter_over, col_name, full_df, pcn_idx,
               get_representative_subset, polar_range, seeds, processes, chunk_size, save_dir, file_name,
               split_per_objective, split_per_bias, split_per_distance, split_per_window, split_per_population,
               skip_subtitle, plot_all, plot_legend_as_subtitles, plot_single_objective):
    # Establish the colour palette
    colour_palette = px.colors.qualitative.Plotly
    if split_per_objective:
        colour_palette = px.colors.qualitative.Plotly
    elif split_per_bias:
        colour_palette = px.colors.qualitative.Plotly_r
    elif split_per_distance:
        colour_palette = px.colors.qualitative.Bold
    elif split_per_bias:
        colour_palette = px.colors.qualitative.Set1
    elif split_per_window:
        colour_palette = px.colors.qualitative.Set2
        if len(iter_over) == 4:
            colour_palette = [colour_palette[3]] + colour_palette[:3]
    elif split_per_population:
        colour_palette = px.colors.qualitative.Set1_r

    cols_titles = None
    n_cols = len(iter_over)
    if plot_single_objective:
        n_cols = 3
        cols_titles = ["Performance", "Group Fairness", "Individual Fairness"]
    elif plot_legend_as_subtitles and not skip_subtitle:
        cols_titles = ["+".join(objs) for objs in requested_objectives] \
            if split_per_objective else list(iter_over.values())

    full_figure = make_subplots(rows=1, cols=n_cols, specs=[[{"type": "polar"} for _ in range(n_cols)]],
                                column_titles=cols_titles)
    font_size = 17
    font_size2 = font_size - 3
    #####
    if plot_single_objective:
        for col, obj_type in enumerate([reward_type, group_type, ind_type]):
            columns = [(f"<span style=\"color:{colour_palette[j]}\">{k}</span>"
                        if TYPE_NOTION[k] == obj_type else k) for j, k in enumerate(all_objectives)]
            full_figure.update_layout({f"polar{col + 1 if col > 0 else ''}": dict(
                radialaxis=dict(visible=True, angle=90, tickfont={"size": font_size2}, range=polar_range),
                angularaxis=dict(categoryarray=columns, rotation=90, tickfont={"size": font_size}))})

    for i, split in enumerate(iter_over):
        o_df = full_df[full_df[col_name] == split]
        print(f"{len(o_df)} non-dominated policies over {len(seeds)} seeds")

        if plot_single_objective:
            print("plot_single")
            columns = [(f"<span style=\"color:{colour_palette[j]}\">{k}</span>"
                        if TYPE_NOTION[k] == TYPE_NOTION[split] else k) for j, k in enumerate(all_objectives)]
        else:
            print("plot")
            reqs = split.split(":") if split_per_objective else requested_objectives[0]
            columns = [(f"<span style=\"color:{colour_palette[i]}\"><b>{k}</b></span>" if k in reqs else k)
                       for j, k in enumerate(all_objectives)]

        if not plot_single_objective:
            full_figure.update_layout({f"polar{i + 1 if i > 0 else ''}": dict(
                radialaxis=dict(visible=True, angle=90, tickfont={"size": font_size2}, range=polar_range),
                angularaxis=dict(categoryarray=columns, rotation=90, tickfont={"size": font_size}))
            })
        if get_representative_subset:
            highlight_indices = find_representative_subset(o_df, all_objectives, all_objectives, seeds,
                                                           processes=processes, chunk_size=chunk_size)
        else:
            highlight_indices = o_df.index

        draw_split = True
        dashes = [None, "dash", "dot", "dashdot"]
        marker_color = colour_palette[0]
        current_dash = 0

        for l, (k, row) in enumerate(o_df.iterrows()):
            r = row.tolist()[:len(all_objectives)] + [row[0]]
            theta = columns + [columns[0]]
            name = "+".join(split) if split_per_objective else iter_over[split]
            draw = True
            showlegend = draw_split and not plot_legend_as_subtitles
            if get_representative_subset:
                name = name
                # Representative policy
                if k in highlight_indices:
                    marker_color = colour_palette[sorted_objectives[split]] if plot_single_objective else \
                        colour_palette[i]
                    opacity = 1.0
                # Other policy
                elif plot_all:
                    marker_color = colour_palette[sorted_objectives[split]]
                    opacity = 0.9 * len(highlight_indices) / len(o_df)
                    showlegend = False
                else:
                    draw = False

            if draw:
                line_width = 2
                col = type_columns[TYPE_NOTION[split]] if plot_single_objective else i + 1
                full_figure.add_trace(go.Scatterpolar(r=r, theta=theta, mode="markers+lines", marker_color=marker_color,
                                                      opacity=opacity, name=name, showlegend=showlegend,
                                                      line_width=line_width, marker_size=line_width * 2.5,
                                                      # line_dash=dashes[current_dash % len(dashes)]
                                                      ), row=1, col=col)
                current_dash += 1
                if showlegend:
                    draw_split = False

    # Update figure
    size = 360
    mm = 20  # 20
    mmm = mm * 1.5
    margins = dict(t=mm, b=mm, l=mmm, r=mmm)
    full_figure.update_layout({"margin": margins, "legend_font_size": font_size2})
    full_figure.update_annotations({"font_size": font_size})  # Subplot titles

    full_figure.write_image(f"{save_dir}/{file_name}{'_' + str(pcn_idx) if pcn_idx not in [-1, None] else ''}.png",
                            width=size * 0.75 * n_cols + margins["r"] * 4,
                            height=size,
                            scale=4)  # 4
    # time.sleep(1)
    # full_figure.write_image(f"{save_dir}/{file_name}{'_' + str(pcn_idx) if pcn_idx not in [-1, None] else ''}.pdf",
    #                         width=size * 0.75 * n_cols + margins["r"] * 4,
    #                         height=size,
    #                         scale=2)  # 4
