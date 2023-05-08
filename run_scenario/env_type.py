import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':  # augmentation, vae (conditional action vae), any latent model. history is not iid: shuffle.
    # vae correlatie tussen features
    # welke scenarios bekijken we
    rng = np.random.default_rng(seed=1)
    n = 500
    y_min = 0
    y_max = 10
    x = np.arange(0, n)

    y_det = np.full_like(x, fill_value=(y_max + y_min) / 2)
    _d = 0.15
    y_stoch = y_det + rng.normal(size=n) * _d - _d / 2

    y_det_nnst = y_det.copy() + _d * 2 * np.sin(x * 0.025)

    xs = rng.choice([i for i in x if 10 < i < n - 10], size=int(rng.uniform(2, 5)))
    xs = sorted(xs)
    x_ind = 0
    delta_a = 1
    delta_b = 0
    da = 0.3
    db = 0.05
    new_ab = []#[(delta_a, delta_b)]
    for _x in xs:
        delta_a *= rng.uniform(1 - da, 1 + da)
        delta_b += rng.uniform(-db, db)
        new_ab.append((delta_a, delta_b))
    for x_ind, _x in enumerate(xs[1:]):
        _da, _db = new_ab[x_ind]
        x0 = 0 if x_ind == 0 else xs[x_ind - 1]
        x1 = -1 if x_ind == len(xs) - 1 else xs[x_ind + 1]
        xvals = np.linspace(x[x0], x[x1], x[x1] - x[x0] + 1)
        y_det_nnst[x0:x1] = y_det_nnst[x0:x1] * _da + _db
        x_ind += 1

    y_stoch_nnst = y_det_nnst + rng.normal(size=n) * 0.2 - 0.1

    ys = [y_det, y_stoch, y_det_nnst, y_stoch_nnst]

    deterministic = "deterministic"
    stochastic = "stochastic"
    nonstationary_det = "nonstationary deterministic"
    nonstationary_stoch = "nonstationary stochastic"
    labels = [deterministic, stochastic, nonstationary_det, nonstationary_stoch]

    colors_style = [("tab:blue", "-"), ("tab:blue", "o"), ("tab:orange", "dashed"), ("tab:orange", "^")]

    fig, ax1 = plt.subplots(figsize=(10, 3))

    plt.title(f"")
    plt.xlabel("timestep $t$")
    plt.ylabel("reward $r$")

    plt.ylim(3.75, 6.75)

    for y, label, (c, s) in zip(ys, labels, colors_style):
        lw = 0.25
        if label in [deterministic, nonstationary_det]:
            plt.plot(x, y, label=label, color=c, ls=s, lw=lw * 15)
        else:
            l = lw if label == stochastic else lw * 0.05
            plt.scatter(x, y, label=label, lw=lw, color=c, marker=s, alpha=0.4)

    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax1.set_xticks([])

    plt.legend()
    plt.tight_layout()
    plt.savefig("deterministic-stochastic-nonstationary.png", dpi=300)
    plt.show()
