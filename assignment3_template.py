import argparse
from pathlib import Path
import numpy as np

SVM_X = ((1, 1), (1, 2), (2, 1), (0, 0), (1, 0), (0, 1))
SVM_Y = (1, 1, 1, -1, -1, -1)


def identify_support_vectors():
    return 1, 5, 6


def derive_svm_separator():
    i = np.array(identify_support_vectors()) - 1
    x, y = np.array(SVM_X)[i], np.array(SVM_Y)[i]
    w = np.linalg.solve(np.c_[x, np.ones(3)], y)
    return tuple(w[:2]), float(w[2]), float(1 / np.linalg.norm(w[:2]))


def plot_svm_data(output_path):
    import matplotlib.pyplot as plt

    x, y = np.array(SVM_X), np.array(SVM_Y)
    f, ax = plt.subplots(figsize=(4, 3.8), layout='constrained')
    for label, c, marker in [(1, 'tab:blue', 'o'), (-1, 'tab:orange', 's')]:
        z = x[y == label]
        ax.scatter(*z.T, c=c, marker=marker, s=45, label=f'{label:+d}')
    for i, z in enumerate(x, 1):
        ax.annotate(str(i), z, xytext=(7, 7), textcoords='offset points')
    i = np.array(identify_support_vectors()) - 1
    ax.scatter(*x[i].T, s=130, facecolors='none', edgecolors='black')
    t = np.linspace(-.5, 2.5, 100)
    ax.plot(t, 1.5-t, c='black')
    ax.plot(t, 1-t, '--', c='gray')
    ax.plot(t, 2-t, '--', c='gray')
    ax.set(xlabel='x1', ylabel='x2', xlim=(-.5, 2.5), ylim=(-.5, 2.5),
           xticks=[0, 1, 2], yticks=[0, 1, 2], aspect='equal')
    ax.legend(fontsize=8, frameon=False)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f.savefig(p, dpi=200)
    plt.close(f)
    return p


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=('plot', 'svm'), default='plot')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'svm_data.png')
    a = p.parse_args()
    if a.task == 'plot':
        print(plot_svm_data(a.output))
    else:
        print('Support vectors:', identify_support_vectors())
        w, b, m = derive_svm_separator()
        print('w:', w, 'b:', b, 'margin:', m)


if __name__ == '__main__':
    main()
