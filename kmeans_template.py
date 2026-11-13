import argparse
from pathlib import Path
import numpy as np

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    x, c = np.asarray(data), np.asarray(centroids)
    return ((x[:, None] - c)**2).sum(axis=2).argmin(axis=1)


def update_centroids(data, centroids, assignments):
    x, a = np.asarray(data), np.asarray(assignments)
    c = np.array(centroids, dtype=float, copy=True)
    for i in range(len(c)):
        z = x[a == i]
        if len(z):
            c[i] = z.mean(axis=0)
    return c


def kmeans(data, centroids, max_iterations=100):
    for _ in range(max_iterations):
        a = update_assignments(data, centroids)
        centroids = update_centroids(data, centroids, a)
    return centroids, update_assignments(data, centroids)


def make_toy_data():
    from sklearn.datasets import make_blobs
    return make_blobs(n_samples=4000, centers=4, cluster_std=.60, random_state=0)


def get_initial_clusters(k, data, rng):
    return data[rng.integers(0, data.shape[0], size=k)].copy()


def run_experiments(output_dir, seed=0, max_iterations=100):
    import matplotlib.pyplot as plt

    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    x, _ = make_toy_data()
    rng = np.random.default_rng(seed)
    out = {}
    for k in K_VALUES:
        c = get_initial_clusters(k, x, rng)
        c, a = kmeans(x, c, max_iterations)
        out[k] = dict(centroids=c, assignments=a)
        f, ax = plt.subplots(figsize=(3, 4), layout='constrained')
        ax.scatter(*x.T, c=a, cmap='tab10', s=3, alpha=.5, edgecolors='none', vmin=0, vmax=9)
        ax.scatter(*c.T, c='black', marker='x', s=40)
        ax.set(xlabel='x1', ylabel='x2', title=f'k={k}', aspect='equal',
               xticks=[-2, 0, 2, 4], yticks=[0, 4, 8])
        f.savefig(p/f'k_{k}.png', dpi=200)
        plt.close(f)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-iterations', type=int, default=100)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'results/kmeans')
    a = p.parse_args()
    run_experiments(a.output_dir, a.seed, a.max_iterations)


if __name__ == '__main__':
    main()
