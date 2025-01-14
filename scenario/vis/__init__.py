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


def load_pcn_log(results_dir, is_fraud, steps, population, bias, distance, window, objectives, compute_objectives, seed,
                 overwrite_dir=None, reverse=False):
    if overwrite_dir:
        exp_dir = f"{overwrite_dir}/seed_{seed}"
    else:
        exp_dir = results_dir if is_fraud else f"{results_dir}/population_{population}/"
        exp_dir += f"/bias_{bias}/steps_{steps}/objectives_{':'.join(objectives)}_{':'.join(compute_objectives)}/" \
                   f"distance_metric_{distance}/window_{window}/seed_{seed}/"
    all_runs = sorted([filename for filename in os.listdir(exp_dir) if filename.startswith("202")], reverse=reverse)
    f = f"{exp_dir}/{all_runs[0]}/pcn_log.csv"
    df = pd.read_csv(f, index_col=None)
    return df


def load_pcn_dataframes(requested_objectives, computed_objectives, all_objectives, sorted_objectives,
                        seeds, steps, pcn_idx, base_results_dir, is_fraud, n_transactions, fraud_proportion, team_size,
                        populations, distances, windows, biases):
    env_name = "fraud_detection" if is_fraud else "job_hiring"
    base_results_dir += f"{env_name}/"
    if is_fraud:
        base_results_dir += f"/n_transactions_{n_transactions}/fraud_proportion_{fraud_proportion}/"
    else:
        base_results_dir += f"/team_{team_size}/"
    results_dir = base_results_dir

    all_dataframes = []
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
                            df = load_pcn_log(results_dir, is_fraud, steps, population, bias, distance,
                                              window, obj, cobj, seed, reverse=len(distances) < 2)
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
    full_df = pd.concat(all_dataframes, ignore_index=True)
    return full_df, results_dir


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
    row, dataframe = args
    return [(j, (row - row2).abs().values) for j, row2 in dataframe.iterrows()]


def get_diffs(dataframe, processes=4, chunk_size=64):
    if len(dataframe) > 350:
        from multiprocessing import Pool
        with Pool(processes=processes) as pool:
            args = [(row1, dataframe) for _, row1 in dataframe.iterrows()]
            differences = pool.map(_diff_pool, args, chunksize=chunk_size)
            pool.close()
            pool.join()
        return differences
    else:
        differences = []
        for _, row1 in dataframe.iterrows():
            diffs = [(j, (row1 - row2).abs().values) for j, row2 in dataframe.iterrows()]
            differences.append(diffs)
    return differences


def find_representative_subset(dataframe, labels, all_objectives, seeds, processes=4, chunk_size=64):
    sorted_o_df = dataframe.sort_values(by=labels, ascending=False)[labels]
    # Compute the extrema for each objective, to find the range of values
    max_objectives = sorted_o_df.max()
    min_objectives = sorted_o_df.min()
    range_obj = max_objectives - min_objectives

    min_improvement = len(all_objectives) - 1  # number of objectives that must differ enough
    highlight_indices = set()
    sums = sorted_o_df.to_numpy(float)
    sums = np.sum(sums, axis=1)
    highlight_indices.add(int(sorted_o_df.iloc[np.argmax(sums)].name))

    differences = get_diffs(sorted_o_df[labels], processes=processes, chunk_size=chunk_size)
    multiplier = 1.0  # 0.5
    delta = 0.5  # 0.05

    best_found = None
    tries = 0
    max_tries = 50
    _min_p = 5
    _max_p = 12

    while (tries <= max_tries) and (len(highlight_indices) < _min_p or len(highlight_indices) > _max_p):
        threshold = range_obj.values * multiplier
        highlight_indices = set()

        for l, (k, row) in enumerate(sorted_o_df.iterrows()):
            if l == 0:
                highlight_indices.add(k)
                keep_indices = [m for m, d in differences[l] if sum(d >= threshold) >= min_improvement]
                highlight_indices.update(keep_indices)
            elif l not in highlight_indices:
                keep_indices = [m for m, d in differences[l] if sum(d >= threshold) >= min_improvement]
                highlight_indices.update(keep_indices)

        if len(highlight_indices) < 4:
            multiplier -= delta * 0.05
        elif len(highlight_indices) > 15:
            multiplier += delta * 0.125
        elif len(highlight_indices) < 7:
            multiplier -= delta * 0.01
        else:
            multiplier += delta * 0.025

        if best_found is None:
            best_found = highlight_indices
        if len(best_found) == 1:
            best_found = highlight_indices

        if len(highlight_indices) == len(best_found):
            tries += 1
        # Previous is much larger, current is better
        elif (len(best_found) - _max_p) > (len(highlight_indices) - _max_p):
            best_found = highlight_indices
            tries = 0
        # Previous is too small, current is better
        elif (_min_p > len(best_found)) and (len(highlight_indices) <= _max_p):
            best_found = highlight_indices
            tries = 0
        else:
            tries += 1
    print(f"{len(highlight_indices)} non-dominated policies kept with threshold {threshold} "
          f"over {len(seeds)} seeds. Keeping best {len(best_found)}")
    return sorted(best_found)
