from enum import Enum

import numpy as np

from scenario.create_fair_env import ALL_OBJECTIVES, OBJECTIVES_MAPPING_r as env_OBJ_MAP_r
from scenario.vis import load_dataframes, get_splits, get_iter_over_save
from scenario.vis.plot import plot_radar, plot_bar, plot_scatter

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    processes = 4  # The number of cores to use when computing the representative sets for high policy counts
    chunk_size = 128  # The chunk size to use when computing the representative sets for high policy counts

    base_results_dir = "/Users/alexandracimpean/Desktop/VSC_Fairness/Nov2024_core/"
    # base_results_dir = "/Users/alexandracimpean/Desktop/VSC_Fairness/May2025/"
    model_name = "PCN"
    # model_name = "DQN"  # TODO
    # Only consider these objectives and (plotting) parameters for plotting and retrieving data
    reduced_objectives = ["R", "SP", "EO", "PE", "OAE", "PP", "IF", "CSC"]
    OBJECTIVES_MAPPING = {env_OBJ_MAP_r[objective]: (objective.name if isinstance(objective, Enum) else objective)
                          for objective in ALL_OBJECTIVES if env_OBJ_MAP_r[objective] in reduced_objectives}
    sorted_objectives = {o: i for i, o in enumerate(reduced_objectives)}
    all_objectives = [o for o in OBJECTIVES_MAPPING]
    #
    # polar_range = [-500, 0]
    # polar_range = [None, None]  # TODO
    polar_ranges = {
        # "job_hiring": [-500, 0],
        # "fraud_detection": [-500, 0],
        "job_hiring": [-300, 0],
        "fraud_detection": [-510, 0],
    }
    max_reward = {  # Theoretical max reward obtainable through environments, for (max) 1000 steps episodes
        "job_hiring": 46.53243,  # Based on empirical runs employing correct action every time to maximise current reward
        "fraud_detection": 906.0,  # Every transaction has been correctly flagged/ignored
    }
    #
    steps = 500000
    ep_length = 1000
    #
    team_size = 100
    n_transactions = 1000
    fraud_proportion = 0.5
    #
    pcn_idx = None
    scaled = False
    #
    get_representative_subset = True
    plot_all = True
    print_repr_policies_table = True
    plot_policies_different_colours = False
    plot_dashed_lines = False
    #
    plot_policies_different_colours = True
    plot_dashed_lines = True

    #
    is_fraud = True
    is_fraud = False
    seeds = range(10)  # TODO 10
    # Single objective
    requested_objectives = [["R"], ["SP"], ["IF"], ["EO"], ["PE"], ["PP"], ["OAE"], ["CSC"]]
    # R_Group_Ind
    # requested_objectives = [["R", "SP", "IF"], ["R", "SP", "CSC"], ["R", "EO", "IF"], ["R", "EO", "CSC"]]
    # R_Group_Ind, windows
    # requested_objectives = [["R", "SP", "IF"]]
    # requested_objectives = [["R"]]

    # Assuming reduced_objectives are computed+optimised ==> the ones not in requested go in computed
    computed_objectives = [[o for o in reduced_objectives if o not in l] for l in requested_objectives]

    # Different distance metrics
    # requested_objectives = [["R", "SP"]]
    # computed_objectives = [["EO", "PE", "OAE", "PP", "IF", "IF", "IF", "CSC", "CSC", "CSC"]]
    # all_objectives = ["R", "SP", "EO", "PE", "OAE", "PP", "IF\\textsubscript{braycurtis}", "IF\\textsubscript{HEOM}", "IF\\textsubscript{HMOM}", "CSC\\textsubscript{braycurtis}", "CSC\\textsubscript{HEOM}", "CSC\\textsubscript{HMOM}"]
    # all_objectives = ["R", "SP", "EO", "PE", "OAE", "PP", "IF_braycurtis", "IF_HEOM", "IF_HMOM", "CSC_braycurtis", "CSC_HEOM", "CSC_HMOM"]
    #
    # Different distance metrics
    # requested_objectives = [["R", "SP", "IF", "IF", "IF"]]
    # computed_objectives = [["EO", "PE", "OAE", "PP"]]
    # all_objectives = ["R", "SP", "EO", "PE", "OAE", "PP", "IF_braycurtis", "IF_HMOM", "IF_HEOM"]
    # #
    # requested_objectives = [["R", "SP", "CSC", "CSC", "CSC"]]
    # computed_objectives = [["EO", "PE", "OAE", "PP"]]
    # all_objectives = ["R", "SP", "EO", "PE", "OAE", "PP", "CSC_braycurtis", "CSC_HMOM", "CSC_HEOM"]
    #

    #
    populations = {
        "belgian_population": "default",
        # "belgian_pop_diff_dist_gen": "gender",
        # "belgian_pop_diff_dist_nat_gen": "nationality-gender",
    }
    distances = {d: d for d in [
        # "braycurtis",
        # "HMOM",
        "HEOM"
        # "braycurtis:HEOM:HMOM:braycurtis:HEOM:HMOM"
        # "braycurtis:HMOM:HEOM"
    ]}
    windows = {w: f"window_{w}" for w in [
        # 100,
        # 200,
        # 500,
        1000,
        # "discount_0.9_0.0001_5_100",
        # "discount_0.95_0.0001_5_100",
        # "discount_1.0_0.0001_5_100",
        # "discount_1.0_0.001_5_100",
        # "discount_1.0_0.0001_5_100",
        # "discount_1.0_0.00001_5_100",
    ]}
    if is_fraud:
        biases = {
            0: "default",
            # 1: r"$+0.1 C_a$",
            # 2: r"$+0.1 C_a m_0$",  # 2: r"$+0.1 C_a merchant_0$",
        }
    else:
        biases = {
            0: "default",
            # 1: "+0.1 men",
            # 2: "+0.1 Belgian men",
        }

    env_name = "fraud_detection" if is_fraud else "job_hiring"
    s_prefix = "s_" if scaled else ""

    #
    requested_objectives = [sorted(l, key=lambda o: sorted_objectives[o]) for l in requested_objectives]
    computed_objectives = [sorted(l, key=lambda o: sorted_objectives[o]) for l in computed_objectives]
    print(requested_objectives)
    print(computed_objectives)

    import time
    ts = time.time()

    #################
    full_df, results_dir, all_pcn_logs = load_dataframes(requested_objectives, computed_objectives,
                                                         all_objectives, sorted_objectives,
                                                         seeds, steps, pcn_idx, base_results_dir,
                                                         is_fraud, n_transactions, fraud_proportion, team_size,
                                                         populations, distances, windows, biases, model_name)
    min_range = min(full_df[all_objectives].min().values)
    full_df[all_objectives[0]] -= max_reward[env_name]  # TODO
    polar_range = polar_ranges[env_name]

    ################
    split_per_objective, split_per_bias, split_per_distance, split_per_window, split_per_population, \
    skip_subtitle, plot_legend_as_subtitles, plot_single_objective = get_splits(env_name, populations, distances,
                                                                                windows, biases, requested_objectives)
    col_name, iter_over, save_dir, file_name = get_iter_over_save(requested_objectives, computed_objectives,
                                                                  populations, distances, windows, biases,
                                                                  results_dir, s_prefix, is_fraud, steps)

    # Plot the radar plot
    plot_radar(requested_objectives, all_objectives, sorted_objectives, iter_over, col_name, full_df, pcn_idx,
               get_representative_subset, polar_range, seeds, processes, chunk_size, save_dir, file_name,
               split_per_objective, split_per_bias, split_per_distance, split_per_window, split_per_population,
               skip_subtitle, plot_all, plot_legend_as_subtitles, plot_single_objective,
               env_name, print_repr_policies_table, plot_policies_different_colours, plot_dashed_lines)

    # plot_bar(requested_objectives, all_objectives, sorted_objectives, iter_over, col_name, full_df, pcn_idx,
    #            get_representative_subset, polar_range, seeds, processes, chunk_size, save_dir, file_name,
    #            split_per_objective, split_per_bias, split_per_distance, split_per_window, split_per_population,
    #            skip_subtitle, plot_all, plot_legend_as_subtitles, plot_single_objective,
    #            env_name, print_repr_policies_table, plot_policies_different_colours, plot_dashed_lines)

    # plot_scatter(requested_objectives, all_objectives, sorted_objectives, iter_over, col_name, all_pcn_logs, pcn_idx,
    #              get_representative_subset, polar_range, seeds, processes, chunk_size, save_dir, file_name,
    #              split_per_objective, split_per_bias, split_per_distance, split_per_window, split_per_population,
    #              skip_subtitle, plot_all, plot_legend_as_subtitles, plot_single_objective,
    #              env_name, print_repr_policies_table, plot_policies_different_colours, plot_dashed_lines)

    te = time.time()
    print(te-ts, "seconds")
    # 51.04523801803589 seconds
    # 32.34789991378784 seconds
