from enum import Enum

import numpy as np

from scenario.create_fair_env import ALL_OBJECTIVES, OBJECTIVES_MAPPING_r as env_OBJ_MAP_r
from scenario.vis import load_pcn_dataframes, get_splits, get_iter_over_save
from scenario.vis.plot import plot_radar

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    processes = 4  # The number of cores to use when computing the representative sets for high policy counts
    chunk_size = 64  # The chunk size to use when computing the representative sets for high policy counts

    base_results_dir = "/Users/alexandracimpean/Desktop/VSC_Fairness/Nov2024_core/"
    # Only consider these objectives and (plotting) parameters for plotting and retrieving data
    reduced_objectives = ["R", "SP", "EO", "PE", "OAE", "PP", "IF", "CSC"]
    OBJECTIVES_MAPPING = {env_OBJ_MAP_r[objective]: (objective.name if isinstance(objective, Enum) else objective)
                          for objective in ALL_OBJECTIVES if env_OBJ_MAP_r[objective] in reduced_objectives}
    sorted_objectives = {o: i for i, o in enumerate(reduced_objectives)}
    all_objectives = [o for o in OBJECTIVES_MAPPING]
    #
    polar_range = [-500, 0]
    max_reward = {  # Theoretical max reward obtainable through environments, for (max) 1000 steps episodes
        "job_hiring": 40,  # Based on empirical runs employing correct action every time to maximise current reward
        "fraud_detection": 1000,  # Every transaction has been correctly flagged/ignored
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
    # requested_objectives = [["R"], ["SP"], ["IF"], ["EO"], ["PE"], ["PP"], ["OAE"], ["CSC"]]
    # R_Group_Ind
    # requested_objectives = [["R", "SP", "IF"], ["R", "SP", "CSC"], ["R", "EO", "IF"], ["R", "EO", "CSC"]]
    # R_Group_Ind, windows
    requested_objectives = [["R", "SP", "IF"]]

    # Assuming reduced_objectives are computed+optimised ==> the ones not in requested go in computed
    computed_objectives = [[o for o in reduced_objectives if o not in l] for l in requested_objectives]

    # Different distance metrics    
    # requested_objectives = [["R", "SP"]]
    # computed_objectives = [["EO", "PE", "OAE", "PP", "IF", "IF", "IF", "CSC", "CSC", "CSC"]]
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
            500,
            # 1000,
            # "500_discount"
        ]}
    if is_fraud:
        biases = {
            0: "default",
            # 1: r"$+0.1 C_a$",
            # 2: r"$+0.1 C_a merchant_0$",
        }
    else:
        biases = {
            0: "default",
            # 1: "+0.1 men",
            # 2: "+0.1 <country> men",
        }

    env_name = "fraud_detection" if is_fraud else "job_hiring"
    s_prefix = "s_" if scaled else ""

    #
    requested_objectives = [sorted(l, key=lambda o: sorted_objectives[o]) for l in requested_objectives]
    computed_objectives = [sorted(l, key=lambda o: sorted_objectives[o]) for l in computed_objectives]
    print(requested_objectives)
    print(computed_objectives)

    #################
    full_df, results_dir = load_pcn_dataframes(requested_objectives, computed_objectives,
                                               all_objectives, sorted_objectives,
                                               seeds, steps, pcn_idx, base_results_dir,
                                               is_fraud, n_transactions, fraud_proportion, team_size,
                                               populations, distances, windows, biases)
    # min_range = min(full_df[all_objectives].min().values)
    full_df[all_objectives[0]] -= max_reward[env_name]

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
