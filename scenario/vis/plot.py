import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
import numpy as np

from scenario.vis import find_representative_subset, reward_type, group_type, ind_type, TYPE_NOTION, type_columns


def is_discount_experiment(names):
    return any([("discount" in i) for i in names.values()])


def get_discount_name(names, is_threshold, split=None):
    split_idx = 3 if is_threshold else 2
    split_name = "threshold" if is_threshold else "discount"
    if split is None:
        return [f"{split_name} {i.split('_')[split_idx]}" for i in names.values()]
    else:
        return f"{split_name} {names[split].split('_')[split_idx]}"


def plot_radar(requested_objectives, all_objectives, sorted_objectives, iter_over, col_name, full_df, pcn_idx,
               get_representative_subset, polar_range, seeds, processes, chunk_size, save_dir, file_name,
               split_per_objective, split_per_bias, split_per_distance, split_per_window, split_per_population,
               skip_subtitle, plot_all, plot_legend_as_subtitles, plot_single_objective,
               env_name, print_repr_policies_table=False, plot_policies_different_colours=False,
               plot_dashed_lines=False):
    # Establish the colour palette
    colour_palette = px.colors.qualitative.Plotly
    table_criteria = "objectives"
    extra_caption = f", when optimising for {'-'.join(requested_objectives[0])}"
    is_discounted = any((isinstance(i, str) and "discount" in i) for i in iter_over)
    is_threshold = is_discounted and any((isinstance(i, str) and "0.00001" in i) for i in iter_over)
    file_name_suffix = ""
    if split_per_bias:
        # colour_palette = px.colors.qualitative.Plotly_r
        table_criteria = "reward biases"
    elif split_per_distance:
        # colour_palette = px.colors.qualitative.Bold
        table_criteria = "individual fairness distance metrics"
    elif split_per_window:
        # colour_palette = px.colors.qualitative.Set2
        table_criteria = "window sizes"
        if is_discounted:
            table_criteria = "discounted histories under different "
            table_criteria += "discount thresholds" if is_threshold else "discount factors"
            file_name_suffix = "_threshold" if is_threshold else "_discount"
        # if len(iter_over) == 4:
        #     colour_palette = [colour_palette[3]] + colour_palette[:3]
    elif split_per_population:
        # colour_palette = px.colors.qualitative.Set1_r
        table_criteria = "populations"
    else:
        extra_caption = ""

    cols_titles = None
    n_cols = len(iter_over)
    table_entries = []
    table_columns = [col_name[0].upper() + col_name[1:], *all_objectives]
    table_summary = []
    table_summary_totals = []
    summary_columns = [col_name[0].upper() + col_name[1:], "Seed", "Statistic", *all_objectives]
    round_number = 5
    if plot_single_objective:
        n_cols = 3
        cols_titles = ["Performance", "Group Fairness", "Individual Fairness"]
    elif plot_legend_as_subtitles and not skip_subtitle:
        if split_per_window and is_discount_experiment(iter_over):
            cols_titles = get_discount_name(iter_over, is_threshold)
        else:
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
            all_objs = [f"{o.split('_')[0]}<sub>{o.split('_')[1]}</sub>" if '_' in o else o for o in all_objectives]
            columns = [(f"<span style=\"color:{'black' if plot_policies_different_colours else colour_palette[i]}\">"
                        f"<b>{k}</b></span>" if k in reqs else k) for j, k in enumerate(all_objs)]

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
        dashes = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot", ]
        marker_color = colour_palette[0]
        current_dash = -1
        current_repr = 0

        # highighlight_df = o_df.loc[highlight_indices]
        highighlight_df = o_df.loc[highlight_indices]
        # print(highighlight_df[all_objectives])

        # TODO
        # colour_palette_ = px.colors.qualitative.Dark2
        # colour_palette_ = [(217, 95, 2), (27, 158, 119), (117, 112, 179), ]
        # colour_palette_ = [(217, 95, 2), (27, 198, 89), (65, 72, 219), ]
        # colour_palette_ = [np.array((230, 97, 0)), np.array((93, 58, 155)), np.array((41, 94, 17))]
        colour_palette_ = [np.array((228, 26, 28)), np.array((77, 175, 74)), np.array((55, 126, 184))]
        # colour_palette_ = [(255, 0, 0), (0, 255, 0), (0, 0, 255), ]
        colour_palette_ = [np.array([r/255, g/255, b/255]) for r, g, b in colour_palette_]
        colour_idx = {o: 0 for o in ["R"]}
        colour_idx.update({o: 1 for o in ["SP", "EO", "OAE", "PP", "PE"]})
        colour_idx.update({o: 2 for o in ["IF", "CSC",
                                          "IF_braycurtis", "IF_HEOM", "IF_HMOM",
                                          "CSC_braycurtis", "CSC_HEOM", "CSC_HMOM"]})

        # colour_palette = {o: colour_palette_[idx] for o, idx in colour_idx.items()}
        # print(colour_palette_)
        # exit()

        def assign_colours(df, colours):
            given_colours = {idx: colour_palette_ for idx in df.index}
            given_weights = {idx: [[], [], []] for idx in df.index}
            given_lines = {idx: "solid" for idx in df.index}  # dash

            maxima = df.max()
            minima = df.min()
            weighted_df = pd.DataFrame(
                {o: (df[o].values - minima[o]) / (maxima[o] - minima[o]) for o in all_objectives},
                index=df.index)
            # print(maxima)
            # print(minima)
            # print(weighted_df)
            # exit()

            for j, row in df.iterrows():
                for o in all_objectives:
                    # if row[o] >= 0.9:
                    #     given_lines[j] = "solid"
                    # elif row[o] >= 0.5 and given_lines[j] != "solid":
                    #     given_lines[j] = "dot"  # ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot", ]
                    o_idx = colour_idx[o]
                    # print(j, o_idx, o)
                    # print(given_weights)
                    given_weights[j][o_idx].append(weighted_df.loc[j][o])
                for k, objectives_weights in enumerate(given_weights[j][:]):
                    # print(given_weights)
                    # print(objectives_weights)
                    given_weights[j][k] = sum(objectives_weights) / len(objectives_weights)

            return given_colours, given_weights, given_lines

        # def mix_colours(p_cols):
        #     rgb_colours = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in p_cols]
        #     n_colours = len(rgb_colours)
        #     mixed_cols = [int(sum(channel) / n_colours) for channel in zip(*rgb_colours)]
        #     mixed_cols = f"#{mixed_cols[0]:02x}{mixed_cols[1]:02x}{mixed_cols[2]:02x}".upper()
        #     return mixed_cols

        # def mix_colours(p_cols, weights):
        #     rgb_colours = np.array([mcolors.to_rgb(c) for c in p_cols])
        #     weights = np.array(weights).reshape(-1, 1)
        #     weights = weights / np.sum(weights)
        #     mixed_cols = np.sum(rgb_colours * weights, axis=0)
        #     return mcolors.to_hex(mixed_cols)

        # def mix_colours(p_cols, weights):
        #     # hsv_colours = np.array([mcolors.rgb_to_hsv(mcolors.to_rgb(c)) for c in p_cols])
        #     hsv_colours = np.array([mcolors.rgb_to_hsv(c) for c in p_cols])
        #     weights = np.array(weights).reshape(-1, 1)
        #     weights = weights / np.sum(weights)
        #     mixed_cols = np.sum(hsv_colours * weights, axis=0)
        #     mixed_cols = mcolors.hsv_to_rgb(mixed_cols)
        #     return mcolors.to_hex(mixed_cols)

        # def mix_colours(p_cols, weights):
        #     mixed_cols = weights[0] * p_cols[0] + weights[1] * p_cols[1] + weights[2] * p_cols[2]
        #     mixed_cols = np.clip(mixed_cols, 0, 1)
        #     # print(mixed_cols)
        #     return mcolors.to_hex(mixed_cols)

        def mix_colours(p_cols, weights):
            # rgb_colours = np.array([mcolors.to_rgb(c) for c in p_cols])
            rgb_colours = np.array(p_cols)
            weights = np.array(weights).reshape(-1, 1)
            weights = weights / np.sum(weights)
            mixed_cols = np.sum(rgb_colours * weights, axis=0)
            return mcolors.to_hex(mixed_cols)

        if not plot_single_objective:
            policy_highlights, policy_weights, policy_lines = assign_colours(highighlight_df, colour_palette)
            # print(policy_highlights)
            # print(policy_weights)
            for p, cs in policy_highlights.items():
                if len(cs) > 0:
                    policy_highlights[p] = mix_colours(cs, policy_weights[p])
                    # print(p, cs, policy_highlights[p])

        # print(policy_highlights)
        # exit()

        for l, (k, row) in enumerate(o_df.iterrows()):
            r = row.tolist()[:len(all_objectives)] + [row[0]]
            theta = columns + [columns[0]]
            # name = "+".join(split) if not (plot_single_objective or split_per_objective) else split
            name = split
            draw = True
            showlegend = draw_split and not plot_legend_as_subtitles
            if get_representative_subset:
                if plot_policies_different_colours and not plot_single_objective:
                    # marker_color = colour_palette[current_repr % len(colour_palette)] if k in highlight_indices else \
                    #     "gray"
                    marker_color = policy_highlights[k] if k in highlight_indices else \
                        "gray"
                else:
                    marker_color = colour_palette[sorted_objectives[split]] if plot_single_objective else \
                        colour_palette[i]
                # Representative policy
                if k in highlight_indices:
                    line_width = 2
                    opacity = 1.0
                    current_repr += 1
                    current_dash += 1
                # Other policy
                elif plot_all:
                    line_width = 1.75
                    opacity = 0.70 * len(highlight_indices) / len(o_df)
                    showlegend = False
                else:
                    draw = False

            if draw:
                if plot_single_objective:
                    line_dash = "solid"  # TODO
                else:
                    line_dash = policy_lines[k] if k in highlight_indices else "solid"
                col = type_columns[TYPE_NOTION[split]] if plot_single_objective else i + 1
                full_figure.add_trace(go.Scatterpolar(r=r, theta=theta, mode="markers+lines", marker_color=marker_color,
                                                      opacity=opacity, name=name, showlegend=showlegend,
                                                      line_width=line_width, marker_size=line_width * 2.5,
                                                      line_dash=line_dash), row=1, col=col)
                if showlegend:
                    draw_split = False

            if print_repr_policies_table and ((not get_representative_subset) or (k in highlight_indices)):
                # Remove trailing zeros after decimal points
                objs = [str(round(o, 1)) if o == int(o) else str(round(o, round_number)) for o in r[:-1]]
                obj_name = ":".join(name.split("_")[1:]) if (split_per_window and "_" in str(name)) else name
                if split_per_population or split_per_bias:
                    obj_name = iter_over[split]
                elif split_per_window and is_discounted:
                    obj_name = get_discount_name(iter_over, is_threshold, split)

                table_entries.append([obj_name, *objs])

        if print_repr_policies_table:
            obj_name = ":".join(name.split("_")[1:]) if (split_per_window and "_" in str(name)) else name
            if split_per_population or split_per_bias:
                obj_name = iter_over[split]
            elif split_per_window and is_discounted:
                obj_name = get_discount_name(iter_over, is_threshold, split)
            # Plot mean and standard deviation per seed and for all seeds
            # Single objective policies only have 1 possible maximum per objective on pareto front
            # ==> no meaningful mean/std per seed
            if not plot_single_objective:
                for seed in seeds:
                    sdf = o_df[o_df["seed"] == seed][all_objectives]
                    smean = sdf.mean()
                    sstd = sdf.std()
                    table_summary.append((obj_name, seed, "mean", *smean.values))
                    table_summary.append((obj_name, seed, "std", *sstd.values))
            # Plot total per experiment
            mean = o_df[all_objectives].mean()
            std = o_df[all_objectives].std()
            table_summary_totals.append((obj_name, "All", "mean", *mean.values))
            table_summary_totals.append((obj_name, "All", "std", *std.values))
    table_summary.extend(table_summary_totals)

    # Update figure
    size = 360
    subscript_names = any([('_' in o) for o in all_objectives])
    mm = 30 if subscript_names else 20  # 20
    mmm = mm * 1.5
    margins = dict(t=mm, b=mm, l=mm if subscript_names else mmm, r=mmm * (2.25 if subscript_names else 1))
    full_figure.update_layout({"margin": margins, "legend_font_size": font_size2})
    full_figure.update_annotations({"font_size": font_size})  # Subplot titles

    full_figure.write_image(
        f"{save_dir}/{file_name}{file_name_suffix}{'_' + str(pcn_idx) if pcn_idx not in [-1, None] else ''}.png",
        width=size * 0.75 * n_cols + margins["l"] * 4,
        height=size,
        scale=4)  # 4
    # time.sleep(1)
    # full_figure.write_image(f"{save_dir}/{file_name}{file_name_suffix}{'_' + str(pcn_idx) if pcn_idx not in [-1, None] else ''}.pdf",
    #                         width=size * 0.75 * n_cols + margins["l"] * 4,
    #                         height=size,
    #                         scale=2)  # 4

    if print_repr_policies_table:
        use_long_table = plot_single_objective or is_discounted or len(table_entries) > 20
        # Policies
        latex_df = pd.DataFrame(table_entries, columns=table_columns)
        latex = latex_df.to_latex(index=False, label=f"table:{env_name.split('_')[0]}{file_name}{file_name_suffix}",
                                  longtable=use_long_table,
                                  caption=f"The representative subset of {' '.join(env_name.split('_'))} policies, for "
                                          f"different {table_criteria}{extra_caption}. "
                                          f"Results rounded to {round_number} decimals.")
        lines = latex.splitlines()
        new_lines = []
        prev_str = None
        for line in lines:
            if "\\\\" in line and "&" in line:
                if prev_str is None:
                    prev_str = line
                elif line.split(" & ")[0] != prev_str.split(" & ")[0]:
                    new_lines.append("\\midrule")
                    prev_str = line
            new_lines.append(line)
        # latex = "\\begin{landscape}\n" + "\n".join(new_lines) + "\n\\end{landscape}"
        latex = "\\begin{scriptsize}\n" + "\n".join(new_lines) + "\n\\end{scriptsize}"
        print("\subsection{" + f"{env_name.split('_')[0]}{file_name}" + "}")
        print(latex)
        print("\n")

        # Summary
        use_long_table = plot_single_objective or is_discounted or len(table_summary) > 20
        latex_df = pd.DataFrame(table_summary, columns=summary_columns)
        latex = latex_df.to_latex(index=False,
                                  label=f"table:{env_name.split('_')[0]}{file_name}{file_name_suffix}_summary",
                                  longtable=use_long_table,
                                  caption=f"The statistical summary of the representative subset of {' '.join(env_name.split('_'))} policies, for "
                                          f"different {table_criteria}{extra_caption}. "
                                          f"Results rounded to {round_number} decimals.")
        lines = latex.splitlines()
        new_lines = []
        prev_str = None
        first_newpage = True
        for line in lines:
            if "\\\\" in line and "&" in line:
                if prev_str is None:
                    prev_str = line
                elif line.split(" & ")[0] != prev_str.split(" & ")[0]:
                    midrule = "\\midrule"
                    # print(line)
                    # print(line.split(" & ")[1], repr(line.split(" & ")[1]))
                    if use_long_table and (
                            line.split(" & ")[1] != "All" or (prev_str.split(" & ")[1] != line.split(" & ")[1])):
                        if first_newpage:
                            first_newpage = False
                        else:
                            midrule = "\\midrule\\newpage"
                    new_lines.append(midrule)
                    prev_str = line
            new_lines.append(line)
        # latex = "\\begin{landscape}\n" + "\n".join(new_lines) + "\n\\end{landscape}"
        latex = "\\begin{scriptsize}\n" + "\n".join(new_lines) + "\n\\end{scriptsize}"
        print("\subsection{" + f"{env_name.split('_')[0]}{file_name}" + "}")
        print(latex)


def plot_bar(requested_objectives, all_objectives, sorted_objectives, iter_over, col_name, full_df, pcn_idx,
             get_representative_subset, polar_range, seeds, processes, chunk_size, save_dir, file_name,
             split_per_objective, split_per_bias, split_per_distance, split_per_window, split_per_population,
             skip_subtitle, plot_all, plot_legend_as_subtitles, plot_single_objective,
             env_name, print_repr_policies_table=False, plot_policies_different_colours=False,
             plot_dashed_lines=False):
    # Establish the colour palette
    colour_palette = px.colors.qualitative.Plotly
    table_criteria = "objectives"
    extra_caption = f", when optimising for {'-'.join(requested_objectives[0])}"
    is_discounted = any((isinstance(i, str) and "discount" in i) for i in iter_over)
    is_threshold = is_discounted and any((isinstance(i, str) and "0.00001" in i) for i in iter_over)
    file_name_suffix = ""
    if split_per_bias:
        colour_palette = px.colors.qualitative.Plotly_r
        table_criteria = "reward biases"
    elif split_per_distance:
        colour_palette = px.colors.qualitative.Bold
        table_criteria = "individual fairness distance metrics"
    elif split_per_window:
        colour_palette = px.colors.qualitative.Set2
        table_criteria = "window sizes"
        if is_discounted:
            table_criteria = "discounted histories under different "
            table_criteria += "discount thresholds" if is_threshold else "discount factors"
            file_name_suffix = "_threshold" if is_threshold else "_discount"
        if len(iter_over) == 4:
            colour_palette = [colour_palette[3]] + colour_palette[:3]
    elif split_per_population:
        colour_palette = px.colors.qualitative.Set1_r
        table_criteria = "populations"
    else:
        extra_caption = ""

    # TODO
    if not split_per_bias:
        colour_palette = px.colors.qualitative.Set2

    cols_titles = None
    n_cols = len(iter_over)
    table_entries = []
    table_columns = [col_name[0].upper() + col_name[1:], *all_objectives]
    table_summary = []
    table_summary_totals = []
    summary_columns = [col_name[0].upper() + col_name[1:], "Seed", "Statistic", *all_objectives]
    round_number = 5
    if plot_single_objective:
        n_cols = 3
        cols_titles = ["Performance", "Group Fairness", "Individual Fairness"]
    elif plot_legend_as_subtitles and not skip_subtitle:
        if split_per_window and is_discount_experiment(iter_over):
            cols_titles = get_discount_name(iter_over, is_threshold)
        else:
            cols_titles = ["+".join(objs) for objs in requested_objectives] \
                if split_per_objective else list(iter_over.values())

    # full_figure = make_subplots(rows=1, cols=n_cols, specs=[[{"type": "polar"} for _ in range(n_cols)]],
    #                             column_titles=cols_titles)
    full_figure = go.Figure()

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
            all_objs = [f"{o.split('_')[0]}<sub>{o.split('_')[1]}</sub>" if '_' in o else o for o in all_objectives]
            columns = [(f"<span style=\"color:{'black' if plot_policies_different_colours else colour_palette[i]}\">"
                        f"<b>{k}</b></span>" if k in reqs else k) for j, k in enumerate(all_objs)]

        if not plot_single_objective:
            # full_figure.update_layout({f"polar{i + 1 if i > 0 else ''}": dict(
            #     radialaxis=dict(visible=True, angle=90, tickfont={"size": font_size2}, range=polar_range),
            #     angularaxis=dict(categoryarray=columns, rotation=90, tickfont={"size": font_size}))
            # })  # TODO
            pass
        if False:  # get_representative_subset: # TODO
            highlight_indices = find_representative_subset(o_df, all_objectives, all_objectives, seeds,
                                                           processes=processes, chunk_size=chunk_size)
        else:
            highlight_indices = o_df.index

        draw_split = True
        dashes = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot", ]
        marker_color = colour_palette[0]
        current_dash = -1
        current_repr = 0

        mean = o_df[all_objectives].mean().to_list()
        std = o_df[all_objectives].std().to_list()

        full_figure.add_trace(go.Bar(
            # name=iter_over[split],  # TODO
            name=iter_over[split].replace("-", "-<br>"),  # TODO
            x=all_objectives, y=mean,
            marker_color=colour_palette[i],
            error_y=dict(type='data', array=std)
        ))

    # Update figure
    size = 360
    subscript_names = any([('_' in o) for o in all_objectives])
    mm = 30 if subscript_names else 20  # 20
    mmm = mm * 1.5
    margins = dict(t=mmm, b=mm, l=mm if subscript_names else mmm, r=mmm * (2.25 if subscript_names else 1))
    full_figure.update_layout({"margin": margins, "legend_font_size": font_size2,
                               "yaxis_title": "outcome", "xaxis_title": "objective",
                               "title": f"Average outcome per objective, given 3 different {'population distributions' if not split_per_bias else 'reward biases'}",
                               })
    full_figure.update_annotations({"font_size": font_size})  # Subplot titles

    full_figure.write_image(
        f"{save_dir}/{file_name}{file_name_suffix}{'_' + str(pcn_idx) if pcn_idx not in [-1, None] else ''}_bar.png",
        width=size * 0.75 * n_cols + margins["l"] * 4,
        height=size * 0.8,
        scale=4)  # 4


def plot_scatter(requested_objectives, all_objectives, sorted_objectives, iter_over, col_name, full_df, pcn_idx,
                 get_representative_subset, polar_range, seeds, processes, chunk_size, save_dir, file_name,
                 split_per_objective, split_per_bias, split_per_distance, split_per_window, split_per_population,
                 skip_subtitle, plot_all, plot_legend_as_subtitles, plot_single_objective,
                 env_name, print_repr_policies_table=False, plot_policies_different_colours=False,
                 plot_dashed_lines=False):
    # Establish the colour palette
    colour_palette = px.colors.qualitative.Plotly
    table_criteria = "objectives"
    extra_caption = f", when optimising for {'-'.join(requested_objectives[0])}"
    is_discounted = any((isinstance(i, str) and "discount" in i) for i in iter_over)
    is_threshold = is_discounted and any((isinstance(i, str) and "0.00001" in i) for i in iter_over)
    file_name_suffix = ""
    if split_per_bias:
        colour_palette = px.colors.qualitative.Plotly_r
        table_criteria = "reward biases"
    elif split_per_distance:
        colour_palette = px.colors.qualitative.Bold
        table_criteria = "individual fairness distance metrics"
    elif split_per_window:
        colour_palette = px.colors.qualitative.Set2
        table_criteria = "window sizes"
        if is_discounted:
            table_criteria = "discounted histories under different "
            table_criteria += "discount thresholds" if is_threshold else "discount factors"
            file_name_suffix = "_threshold" if is_threshold else "_discount"
        if len(iter_over) == 4:
            colour_palette = [colour_palette[3]] + colour_palette[:3]
    elif split_per_population:
        colour_palette = px.colors.qualitative.Set1_r
        table_criteria = "populations"
    else:
        extra_caption = ""

    cols_titles = None
    n_cols = len(iter_over)
    table_entries = []
    table_columns = [col_name[0].upper() + col_name[1:], *all_objectives]
    table_summary = []
    table_summary_totals = []
    summary_columns = [col_name[0].upper() + col_name[1:], "Seed", "Statistic", *all_objectives]
    round_number = 5
    if plot_single_objective:
        n_cols = 3
        cols_titles = ["Performance", "Group Fairness", "Individual Fairness"]
    elif plot_legend_as_subtitles and not skip_subtitle:
        if split_per_window and is_discount_experiment(iter_over):
            cols_titles = get_discount_name(iter_over, is_threshold)
        else:
            cols_titles = ["+".join(objs) for objs in requested_objectives] \
                if split_per_objective else list(iter_over.values())

    # full_figure = make_subplots(rows=1, cols=n_cols, specs=[[{"type": "polar"} for _ in range(n_cols)]],
    #                             column_titles=cols_titles)
    full_figure = go.Figure()

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
            all_objs = [f"{o.split('_')[0]}<sub>{o.split('_')[1]}</sub>" if '_' in o else o for o in all_objectives]
            columns = [(f"<span style=\"color:{'black' if plot_policies_different_colours else colour_palette[i]}\">"
                        f"<b>{k}</b></span>" if k in reqs else k) for j, k in enumerate(all_objs)]

        draw_split = True
        dashes = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot", ]
        marker_color = colour_palette[0]
        current_dash = -1
        current_repr = 0

        full_df["R"] = [float(e[2:-2]) for e in full_df["nd_coverage_set"].values]
        groups = full_df.groupby("seed")

        mean = groups["R"].mean()  # .to_list()
        std = groups["R"].std()  # .to_list()

        eps = []

        for _, g in groups:
            # print(g)
            eps = g["episode"]

        # print(groups["episode"])

        # print(iter_over, split)

        # full_figure = px.line(full_df, x="t", y="R", color="seed")
        full_figure.add_trace(go.Scatter(
            # name="seed",
            mode="lines",
            x=eps,  # groups["episode"],
            y=mean,
            error_y=dict(type='data', array=std)
        ))
        full_figure.update_layout(dict(yaxis_range=[0, 950 if env_name == "fraud" else 50]))

    # Update figure
    size = 360
    subscript_names = any([('_' in o) for o in all_objectives])
    mm = 30 if subscript_names else 20  # 20
    mmm = mm * 1.5
    margins = dict(t=mm, b=mm, l=mm if subscript_names else mmm, r=mmm * (2.25 if subscript_names else 1))
    full_figure.update_layout({"margin": margins, "legend_font_size": font_size2})
    full_figure.update_annotations({"font_size": font_size})  # Subplot titles

    full_figure.write_image(
        f"{save_dir}/{file_name}{file_name_suffix}{'_' + str(pcn_idx) if pcn_idx not in [-1, None] else ''}_scatter.png",
        width=size * 0.75 * n_cols + margins["l"] * 4,
        height=size,
        scale=4)  # 4


def plot_scatter_all(full_dfs, save_dir, file_name):
    full_figure = make_subplots(rows=1, cols=2, column_titles=["Job Hiring", "Fraud Detection"])
    # full_figure = go.Figure()

    font_size = 17
    font_size2 = font_size - 3
    colour_palette = px.colors.qualitative.Plotly
    colours = {"PCN": colour_palette[0], "DQN": colour_palette[1]}
    #####
    # time_scale = "t"
    time_scale = "episode"
    axis_title = "Timestep" if time_scale == "t" else "Episode"

    def _get_rewards(df):
        df["R"] = [float(e[2:-2]) for e in df["nd_coverage_set"].values]
        groups = df.groupby([time_scale])
        mean = groups["R"].mean()  # .to_list()
        std = groups["R"].std()  # .to_list()
        for _, g in groups:
            eps = g[time_scale]
            break
        eps = df[df["seed"] == 0][time_scale]
        return mean, std, eps

    def _get_rewards_per_seed(df):
        df["R"] = [float(e[2:-2]) for e in df["nd_coverage_set"].values]
        groups = df.groupby("seed")
        return groups

    def _get_avg_rewards_per_seed(df):
        import numpy as np
        all_rs = [[np.fromstring(s, sep=" ", dtype=np.float32)[0] for s in v.replace("\n", "")[2:-2].split("] [")]
                  for v in df["coverage_set"].values]
        df["means"] = [np.nanmean(r) for r in all_rs]  # TODO
        df["stds"] = [np.nanstd(r) for r in all_rs]
        return df

    name_added = ["PCN"]  # TODO
    for name, is_fraud, full_df in full_dfs:
        # print(name, max([float(e[2:-2]) for e in full_df["nd_coverage_set"].values]))
        mean, std, xs = _get_rewards(full_df)
        import numpy as np
        x = xs.tolist()
        mean = np.array(mean)
        std = np.nan_to_num(np.array(std))
        full_figure.add_trace(go.Scatter(
            name=name,
            mode="lines",
            x=x,
            y=mean,
            marker_color=colours[name],
            showlegend=name not in name_added,
        ), row=1, col=2 if is_fraud else 1)
        y_upper = mean + std
        y_lower = mean - std
        # bound requires explicit ordering, ensure this is the case
        x_sorted_indices = np.argsort(x)
        y_upper = y_upper[x_sorted_indices]
        y_lower = y_lower[x_sorted_indices]

        full_figure.add_trace(
            go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([y_upper, y_lower[::-1]]),
                       fill='toself', fillcolor=colours[name], line_color=colours[name], opacity=0.3, showlegend=False),
            row=1, col=2 if is_fraud else 1)
        name_added.append(name)

        # groups = _get_rewards_per_seed(full_df)
        # for g, group in groups:
        #     full_figure.add_trace(go.Scatter(
        #         name=name,
        #         mode="lines",
        #         x=group[time_scale],
        #         y=group["R"],
        #         marker_color=colours[name],
        #         showlegend=name not in name_added,
        #     ), row=1, col=2 if is_fraud else 1)
        #     name_added.append(name)

        # full_df = _get_avg_rewards_per_seed(full_df)
        # groups = full_df.groupby("seed")
        # for g, group in groups:
        #     full_figure.add_trace(go.Scatter(
        #         name=name,
        #         mode="lines",
        #         x=group[time_scale],
        #         y=group["means"],
        #         marker_color=colours[name],
        #         showlegend=name not in name_added,
        #         # error_y=dict(type='data', array=group["stds"])
        #     ), row=1, col=2 if is_fraud else 1)
        #     name_added.append(name)

    full_figure.update_layout(dict(  # yaxis_range=[0, 45], yaxis2_range=[600, 900],
        yaxis_title_text="Performance reward", xaxis_title_text=axis_title,
        yaxis2_title2_text="Performance reward", xaxis2_title_text=axis_title,
    ))

    # Update figure
    size = 360
    mm = 20
    mmm = mm * 1.5
    margins = dict(t=mm, b=mm, l=mmm, r=mmm)
    full_figure.update_layout({"margin": margins, "legend_font_size": font_size2})
    full_figure.update_annotations({"font_size": font_size})  # Subplot titles

    full_figure.write_image(
        f"{save_dir}/{file_name}_performance.png",
        # width=size * 2.1,
        # height=size,
        width=size * 2.5,
        height=size * 0.48,  # height=size * 0.7,
        scale=4)  # 4
