"""CSE 242 Assignment 3, Question 1: hard-margin linear SVM (30 points).
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np


SVM_X = ((1, 1), (1, 2), (2, 1), (0, 0), (1, 0), (0, 1))
SVM_Y = (1, 1, 1, -1, -1, -1)


def plot_svm_data(output_path):
    """Generate an annotated scatter plot to help inspect Q1(a) by eye.

    Point IDs are 1-based and match the source table order. This helper
    does not identify support vectors or draw the separating boundary.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib could not be imported. Use the README's clean virtual "
            "environment and install requirements.txt."
        ) from exc

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, color, marker in ((1, "tab:blue", "o"), (-1, "tab:orange", "s")):
        points = [x for x, y in zip(SVM_X, SVM_Y) if y == label]
        ax.scatter(
            [x[0] for x in points], [x[1] for x in points],
            color=color, marker=marker, s=90, label=f"y = {label:+d}",
        )
    for point_id, (x1, x2) in enumerate(SVM_X, start=1):
        ax.annotate(str(point_id), (x1, x2), xytext=(7, 7),
                    textcoords="offset points")
    ax.set(xlabel="x1", ylabel="x2", title="Assignment 3: SVM training data",
           xlim=(-0.5, 2.5), ylim=(-0.5, 2.5))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def identify_support_vectors(x_data=None, y_data=None):
    """Q1(a), 10 points together with the plot: identify support vectors by eye.

    Return their 1-based point IDs and explain your choice in the report.
    """
    x = np.asarray(SVM_X if x_data is None else x_data, dtype=float)
    y = np.asarray(SVM_Y if y_data is None else y_data, dtype=float)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("x_data must be 2D and y_data must match its rows.")
    if not np.all(np.isin(y, (-1, 1))):
        raise ValueError("Labels must be -1 or +1.")

    candidates = []
    for active_indices in combinations(range(len(x)), x.shape[1] + 1):
        design = np.column_stack((x[list(active_indices)], np.ones(x.shape[1] + 1)))
        try:
            solution = np.linalg.solve(design, y[list(active_indices)])
        except np.linalg.LinAlgError:
            continue
        weights, bias = solution[:-1], solution[-1]
        signed_distances = y * (x @ weights + bias)
        if np.all(signed_distances >= 1.0 - 1e-10):
            candidates.append((np.linalg.norm(weights), weights, bias))
    if not candidates:
        raise ValueError("The training data is not linearly separable.")

    _, weights, bias = min(candidates, key=lambda candidate: candidate[0])
    signed_distances = y * (x @ weights + bias)
    support_indices = np.flatnonzero(np.isclose(signed_distances, 1.0, atol=1e-10))
    return tuple(int(index + 1) for index in support_indices)


def derive_svm_separator(x_data=None, y_data=None):
    """Q1(b), 10 points for the boundary and 10 for the geometric margin.

    Use a linear, hard-margin SVM. The decision boundary is w dot x + b = 0;
    retain the intercept b. Use the support vectors to derive it.

    Return (weights, bias, geometric_margin), with weights a pair of numbers.
    Explain the derivation and your margin convention in the written report.
    """
    x = np.asarray(SVM_X if x_data is None else x_data, dtype=float)
    y = np.asarray(SVM_Y if y_data is None else y_data, dtype=float)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("x_data must be 2D and y_data must match its rows.")
    if not np.all(np.isin(y, (-1, 1))):
        raise ValueError("Labels must be -1 or +1.")

    # At the optimum, d + 1 active constraints determine d weights and b.
    candidates = []
    active_count = x.shape[1] + 1
    for active_indices in combinations(range(len(x)), active_count):
        design = np.column_stack((x[list(active_indices)], np.ones(active_count)))
        try:
            solution = np.linalg.solve(design, y[list(active_indices)])
        except np.linalg.LinAlgError:
            continue

        weights, bias = solution[:-1], solution[-1]
        signed_distances = y * (x @ weights + bias)
        if np.all(signed_distances >= 1.0 - 1e-10):
            candidates.append((np.linalg.norm(weights), weights, bias))

    if not candidates:
        raise ValueError("The training data is not linearly separable.")

    _, weights, bias = min(candidates, key=lambda candidate: candidate[0])
    geometric_margin = 1.0 / np.linalg.norm(weights)
    return tuple(float(weight) for weight in weights), float(bias), float(geometric_margin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plot", "svm"),
                        default="plot", help="Default: generate the SVM data plot.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "svm_data.png",
                        help="Output image path for --task plot.")
    args = parser.parse_args()
    if args.task == "plot":
        print(f"Saved plot to {plot_svm_data(args.output)}")
    else:
        print("Support vector point IDs:", identify_support_vectors())
        weights, bias, margin = derive_svm_separator()
        print(f"Weights: {weights}; bias: {bias}; geometric margin: {margin}")


if __name__ == "__main__":
    main()
