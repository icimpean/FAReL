


if __name__ == '__main__':
    # Namespace(objectives=[0, 1, 6], env='job', lr=0.01, steps=100000.0, batch=128, model_updates=20, top_episodes=10,
    # n_episodes=10, er_size=100, threshold=0.02, noise=0.0, model='densesmall', seed=0, vsc=1,
    # team_size=20, episode_length=100, diversity_weight=0,
    # population='scenario/job_hiring/data/belgian_population.csv', window=100, fair_alpha=0.1)

    seeds = range(10)
    environments = ["job", "fraud"]

    objectives = {"job": [0, 1, 2, 5], "fraud": [0, 3, 4, 6]}
    num_steps = {"job": 100000, "fraud": 500000}

    # Hiring
    populations = {"job": ["belgian_population", "belgian_pop_diff_dist_gen", "belgian_pop_diff_dist_nat_gen"]}
    team_sizes_eps_length = {"job": [(20, 200), (100, 1000)]}
    diversity_weights = {"job": [0]}

    # Fraud
    # /
    populations["fraud"] = ["default"]
    team_sizes_eps_length["fraud"] = [(0, 7)]
    diversity_weights["fraud"] = [0]

    # Fairness framework
    window = 100
    fair_alpha = 0.1

    parameters = []
    for env in environments:
        for seed in seeds:
            for pop in populations[env]:
                for (size, length) in team_sizes_eps_length[env]:
                    for div_w in diversity_weights[env]:
                        obj = objectives[env]
                        steps = num_steps[env]
                        entry = {"env": env, "seed": seed, #"objectives": obj,
                                 "steps": steps,
                                 "team_size": size, "episode_length": length, "diversity_weight": div_w,
                                 "population": pop, "window": window, "fair_alpha": fair_alpha}
                        parameters.append(entry)

                        entry = entry.copy()
                        entry["seed"] = "${i}"
                        text = ",".join([f"{k}=" + (str(v) if (not isinstance(v, str)) or k == "seed" else f'\"{v}\"') for k, v in entry.items()])
                        print(text)

                        # break
                    # break
                # break
            break
    import csv

    path = "fair_configs2.csv"
    with open(path, mode="w") as file:
        writer = csv.DictWriter(file, fieldnames=parameters[0].keys())
        writer.writeheader()
        writer.writerows(parameters)
