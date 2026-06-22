import os

import numpy as np
import pandas as pd

from fairness.group import GroupNotion
from fairness.individual import IndividualNotion
from scenario.create_fair_env import OBJECTIVES_MAPPING as env_OBJ_MAP

group_type = "G"
ind_type = "I"
reward_type = "R"
type_columns = {reward_type: 1, group_type: 2, ind_type: 3}
TYPE_NOTION = {abbrev: (group_type if isinstance(notion, GroupNotion) else
                        ind_type if isinstance(notion, IndividualNotion) else reward_type)
               for abbrev, notion in env_OBJ_MAP.items()}


def get_id(*args):
    return "_".join([str(a) for a in args])


def load_agent_log(results_dir, is_fraud, steps, population, bias, distance, window, objectives, compute_objectives,
                   seed, overwrite_dir=None, reverse=False, name="PCN"):
    if overwrite_dir:
        exp_dir = f"{overwrite_dir}/seed_{seed}"
    else:
        exp_dir = results_dir if is_fraud else f"{results_dir}/population_{population}/"
        exp_dir += f"/bias_{bias}/steps_{steps}/objectives_{':'.join(objectives)}_{':'.join(compute_objectives)}/" \
                   f"distance_metric_{distance}/window_{window}/seed_{seed}/"
    all_runs = sorted([filename for filename in os.listdir(exp_dir) if filename.startswith("202")], reverse=reverse)
    f = f"{exp_dir}/{all_runs[0]}/{name.lower()}_log.csv"
    df = pd.read_csv(f, index_col=None)
    return df


def load_dataframes(requested_objectives, computed_objectives, all_objectives, sorted_objectives,
                    seeds, steps, pcn_idx, base_results_dir, is_fraud, n_transactions, fraud_proportion, team_size,
                    populations, distances, windows, biases, name="PCN"):
    env_name = "fraud_detection" if is_fraud else "job_hiring"
    base_results_dir += f"{env_name}/"
    if is_fraud:
        base_results_dir += f"/n_transactions_{n_transactions}/fraud_proportion_{fraud_proportion}/"
    else:
        base_results_dir += f"/team_{team_size}/"
    results_dir = base_results_dir

    all_dataframes = []
    all_pcn_logs = []
    for population, population_name in populations.items():
        for bias, bias_name in biases.items():
            for distance, distance_name in distances.items():
                for window, window_name in windows.items():
                    for obj, cobj in zip(requested_objectives, computed_objectives):
                        objectives_indices = [sorted_objectives[o] for o in obj]
                        is_single = len(obj) == 1
                        pcn_logs = []
                        # Load in all seeds per experiment
                        for seed in seeds:
                            print("Exp.", population, bias, distance, window, obj, cobj, seed, f"is_single={is_single}")
                            df = load_agent_log(results_dir, is_fraud, steps, population, bias, distance,
                                                window, obj, cobj, seed, reverse=len(distances) < 2, name=name)
                            pcn_logs.append(df)
                        # Get coverage sets
                        cs, ndcs = get_coverage_sets(pcn_logs, objectives_indices, is_single, idx=pcn_idx)
                        dfs = [pd.DataFrame(s, columns=all_objectives) for s in ndcs]
                        for seed, df in zip(seeds, dfs):
                            df["seed"] = seed
                            df["population"] = population
                            df["bias"] = bias
                            df["distance"] = distance
                            df["window"] = window
                            df["objectives"] = ":".join(obj)
                            df_id = get_id(population, bias, distance, window, requested_objectives, seed)
                            df["id"] = df_id
                        df_ndcs = pd.concat(dfs, ignore_index=True)
                        all_dataframes.append(df_ndcs)
                        for seed, df in zip(seeds, pcn_logs):
                            df["seed"] = seed
                            df["population"] = population
                            df["bias"] = bias
                            df["distance"] = distance
                            df["window"] = window
                            df["objectives"] = ":".join(obj)
                            df_id = get_id(population, bias, distance, window, requested_objectives, seed)
                            df["id"] = df_id
                        pcn_logs = pd.concat(pcn_logs, ignore_index=True)
                        all_pcn_logs.append(pcn_logs)
    full_df = pd.concat(all_dataframes, ignore_index=True)
    all_pcn_logs = pd.concat(all_pcn_logs, ignore_index=True)
    return full_df, results_dir, all_pcn_logs


def get_splits(env_name, populations, distances, windows, biases, requested_objectives):
    split_per_population = env_name == "job_hiring" and len(populations) > 1
    split_per_bias = len(biases) > 1
    split_per_distance = len(distances) > 1
    split_per_window = len(windows) > 1

    plot_single_objective = all([len(o) == 1 for o in requested_objectives])
    split_per_objective = plot_single_objective or not any([split_per_population, split_per_bias,
                                                            split_per_distance, split_per_window])
    plot_legend_as_subtitles = True
    skip_subtitle = False

    # Currently, only one type of split is supported
    all_splits = (split_per_objective, split_per_bias, split_per_distance, split_per_window, split_per_population)
    assert sum(all_splits) < 2, f"Only one type of split allowed for a plot, given: {all_splits}. "

    return split_per_objective, split_per_bias, split_per_distance, split_per_window, split_per_population, \
           skip_subtitle, plot_legend_as_subtitles, plot_single_objective


def get_iter_over_save(requested_objectives, computed_objectives, populations, distances, windows, biases,
                       results_dir, s_prefix, is_fraud, steps):
    env_name = "fraud_detection" if is_fraud else "job_hiring"
    save_dir = results_dir

    split_per_objective, split_per_bias, split_per_distance, split_per_window, split_per_population, \
    skip_subtitle, plot_legend_as_subtitles, plot_single_objective = get_splits(env_name, populations, distances,
                                                                                windows, biases, requested_objectives)

    _population = [p for p in populations][0]
    _bias = [p for p in biases][0]
    _distance = [p for p in distances][0]
    _window = [p for p in windows][0]
    _n = [".".join(o) for o in requested_objectives]
    _obj = "-".join(_n if plot_single_objective else _n[:1])
    obj = "-".join([":".join(o) for o in requested_objectives][:1])
    cobj = "-".join([":".join(o) for o in computed_objectives][:1])

    file_name = f"{s_prefix}{_obj}_b{_bias}_d{_distance}_w{_window}"
    if not is_fraud and len(populations) == 1:
        save_dir = f"{results_dir}/population_{_population}/"
    if split_per_objective:
        col_name = "objectives"
        iter_over = [":".join(obj) for obj in requested_objectives]
        save_dir += f"/bias_{_bias}/steps_{steps}/"
    elif split_per_bias:
        col_name = "bias"
        iter_over = biases
        file_name = f"{s_prefix}{_obj}_b{''.join([str(b) for b in biases])}_d{_distance}_w{_window}"
    elif split_per_distance:
        col_name = "distance"
        iter_over = distances
        save_dir += f"/bias_{_bias}/steps_{steps}/objectives_{obj}_{cobj}/"
        file_name = f"{s_prefix}{_obj}_b{_bias}_distances_w{_window}"
    elif split_per_window:
        col_name = "window"
        iter_over = windows
        save_dir += f"/bias_{_bias}/steps_{steps}/objectives_{obj}_{cobj}/distance_metric_{_distance}/"
        file_name = f"{s_prefix}{_obj}_b{_bias}_d{_distance}_windows"
    elif split_per_population:
        col_name = "population"
        iter_over = populations
    else:
        raise RuntimeError

    return col_name, iter_over, save_dir, file_name


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


def _last_coverage_set(df, name):
    return df[name].tail(1).values[0]  # At end of training


def _read_2d_array(df, name, idx=None):
    if idx is None:
        string = _last_coverage_set(df, name)
    else:
        string = df[name].iloc[idx]
        print(idx, df[name].iloc[idx])
    new_string = string.replace("\n", "")[2:-2].split("] [")
    array = np.vstack([np.fromstring(s, sep=" ", dtype=np.float32) for s in new_string])
    return array


def get_coverage_sets(pcn_dfs, objectives_indices, is_single, idx=None):
    coverage_sets = []
    nd_coverage_sets = []
    print("objectives_indices", objectives_indices)

    for pi, pcn_df in enumerate(pcn_dfs):
        try:
            coverage_set = _read_2d_array(pcn_df, "coverage_set", idx)
            nd_coverage_set = _read_2d_array(pcn_df, "nd_coverage_set", idx)
        except IndexError as e:
            full_nd_coverage_set = np.zeros(shape=(1, len(objectives_indices)))
            # coverage_sets.append(coverage_set)
            nd_coverage_sets.append(full_nd_coverage_set)
            continue

        len_objs = coverage_set.shape[1]
        full_nd_coverage_set = np.zeros(shape=(len(nd_coverage_set), len_objs))
        for i, nd in enumerate(nd_coverage_set):
            rn = 6
            filter = np.round(coverage_set[:, objectives_indices], rn) == np.round(nd, rn)
            try:
                if is_single:
                    j = np.argwhere(filter)
                    j = [np.argmax(filter)] if len(j) == 0 else j[0, 0]
                else:
                    j = np.argwhere(np.all(filter, axis=1))
                    j = np.argmax(np.sum(filter, axis=1)) if len(j) == 0 else j[0]
                cov_set = coverage_set[j]
                idc = len(cov_set) if cov_set.ndim == 1 else len(cov_set[0])
                full_nd_coverage_set[i, :idc] = cov_set
            except IndexError as e:
                print(coverage_set)
                print(nd)
                print(filter)
                print(e)
                raise e

        coverage_sets.append(coverage_set)
        nd_coverage_sets.append(full_nd_coverage_set)

    return coverage_sets, nd_coverage_sets


def _diff_pool(args):
    row, dataframe, _ = args
    return [(j, (row - row2).abs().values) for j, row2 in dataframe.iterrows()]


def _diff_pool_highlight(args):
    row, dataframe, highlighted = args
    return [(j, (row - row2).abs().values) for j, row2 in dataframe.iterrows() if j in highlighted]


def get_diffs(dataframe, processes=4, chunk_size=64, highlighted=None):
    if len(dataframe) > 350:
        from multiprocessing import Pool
        with Pool(processes=processes) as pool:
            args = [(row1, dataframe, highlighted) for _, row1 in dataframe.iterrows()]
            differences = pool.map(_diff_pool_highlight if highlighted else _diff_pool, args, chunksize=chunk_size)
            pool.close()
            pool.join()
        return differences
    else:
        differences = []
        for _, row1 in dataframe.iterrows():
            diffs = [(j, (row1 - row2).abs().values) for j, row2 in dataframe.iterrows()
                     if (not highlighted) or (j in highlighted)]
            differences.append(diffs)
    return differences


def find_representative_subset(dataframe, labels, all_objectives, seeds, processes=4, chunk_size=64):
    # sorted_o_df = dataframe.sort_values(by=labels, ascending=False)[labels]
    sorted_o_df = dataframe[labels]

    maxima = {o: [] for o in all_objectives}
    for o in all_objectives:
        o_max = sorted_o_df[o].max()
        all_o_max = sorted_o_df[sorted_o_df[o] == o_max].index
        maxima[o] = [m for m in all_o_max]
    # Keep as few policies as possible which contain the maxima for all objectives
    p_counts = {}
    kept_policies = set()
    for o, policies in maxima.items():
        if len(policies) == 1:
            kept_policies.add(policies[0])

    for o, policies in maxima.items():
        for p in policies:
            if p in kept_policies:
                continue
            elif p_counts.get(p):
                p_counts[p] += 1
            else:
                p_counts[p] = 1
    # print("maxima", maxima)
    # print("kept_policies", kept_policies)
    # print("p_counts", p_counts)

    for o, policies in maxima.items():
        if [(p in kept_policies) for p in policies]:
            continue
        pc = [p_counts[p] for p in policies]
        pmax = np.argmax(pc)
        kept_policies.add(policies[pmax])

    differences = get_diffs(sorted_o_df[labels], processes=processes, chunk_size=chunk_size, highlighted=kept_policies)
    # Get differences of all remaining indices with kept policies so far
    reduced_diffs = {}
    for i, diffs in enumerate(differences):
        idx_i = sorted_o_df.iloc[i].name
        if idx_i in kept_policies:
            continue
        reduced_diffs[idx_i] = np.zeros(len(labels))
        for j, diff in diffs:
            # print(j)
            idx_j = j #sorted_o_df.loc[j].name  # j
            if idx_j in kept_policies:
                # reduced_diffs[idx_i] += np.sum(diff)
                reduced_diffs[idx_i] += diff
    print(kept_policies)
    # print(len(differences[0]))
    # print(len(reduced_diffs[0]))
    # print(reduced_diffs[0])

    # Keep the policies with the greatest differences to the kept policies
    # reduced_diffs = sorted(reduced_diffs.items(), key=lambda item: item[1], reverse=True)
    # reduced_diffs = sorted(reduced_diffs.items(), key=lambda item: np.sum(item[1]), reverse=True)
    # reduced_diffs = sorted(reduced_diffs.items(), key=lambda item: np.mean(item[1]), reverse=True)
    # reduced_diffs = sorted(reduced_diffs.items(), key=lambda item: np.std(item[1]), reverse=True)
    # reduced_diffs = sorted(reduced_diffs.items(), key=lambda item: np.max(item[1]), reverse=True)
    # reduced_diffs = sorted(reduced_diffs.items(), key=lambda item: -np.min(item[1]), reverse=True)
    # reduced_diffs = sorted(reduced_diffs.items(), key=lambda item: np.mean(item[1]) - np.std(item[1]), reverse=True)
    reduced_diffs = sorted(reduced_diffs.items(), key=lambda item: np.mean(item[1]) - 0.25 * np.std(item[1]), reverse=True)
    # print(list(reduced_diffs))

    _min_p = 10  # 5, 8
    n_sample = _min_p - len(kept_policies)
    if n_sample > 0:
        kept_policies.update([p for p, diff in reduced_diffs[:n_sample]])

    print(kept_policies)
    # exit()

    return sorted(kept_policies)
