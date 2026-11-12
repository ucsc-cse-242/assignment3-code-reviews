"""CSE 242 Assignment 3, Question 1: hard-margin linear SVM (30 points).
"""

import argparse
from pathlib import Path


SVM_X = ((1, 1), (1, 2), (2, 1), (0, 0), (1, 0), (0, 1))
SVM_Y = (1, 1, 1, -1, -1, -1)

def plot_svm_data(output_path):
    """Annotated scatter plot of the SVM data with the separator and margins.

    Point IDs are 1-based and match the source table order. Support vectors
    come from identify_support_vectors() and are drawn as stars. The boundary
    and margin lines come from derive_svm_separator().
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib could not be imported. Use the README's clean virtual "
            "environment and install requirements.txt."
        ) from exc

    sv_ids = set(identify_support_vectors())
    (w1, w2), b, geo_margin = derive_svm_separator()
    w_norm = (w1 ** 2 + w2 ** 2) ** 0.5

    fig, ax = plt.subplots(figsize=(6, 6))

    # Regular points (class markers), skipping support vectors
    for label, color, marker in ((1, "tab:blue", "o"), (-1, "tab:orange", "s")):
        pts = [x for i, (x, y) in enumerate(zip(SVM_X, SVM_Y), start=1)
               if y == label and i not in sv_ids]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   color=color, marker=marker, s=90, label=f"y = {label:+d}")

    # Support vectors: larger stars in their class color
    for label, color in ((1, "tab:blue"), (-1, "tab:orange")):
        pts = [x for i, (x, y) in enumerate(zip(SVM_X, SVM_Y), start=1)
               if y == label and i in sv_ids]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                       color=color, marker="*", s=300,
                       edgecolors="black", linewidths=1.2, zorder=3)
    # One legend entry for all support vectors
    ax.scatter([], [], color="white", marker="*", s=200,
               edgecolors="black", label="support vector")

    # Point ID labels
    for point_id, (x1, x2) in enumerate(SVM_X, start=1):
        ax.annotate(str(point_id), (x1, x2), xytext=(7, 7),
                    textcoords="offset points")

    # Boundary (w.x + b = 0) and margin lines (w.x + b = +/- geo_margin * ||w||)
    offset = geo_margin * w_norm   # equals 1 under the +/-1 convention
    xs = np.array([-0.5, 2.5])

    def line_points(c):
        """Points on the line w1*x1 + w2*x2 + b = c, across the plot range."""
        if w2 != 0:
            return xs, (c - b - w1 * xs) / w2
        return np.full(2, (c - b) / w1), np.array([-0.5, 2.5])

    x_b, y_b = line_points(0)
    ax.plot(x_b, y_b, "k-", linewidth=2, label="decision boundary")
    for c, name in ((offset, "margin"), (-offset, None)):
        x_m, y_m = line_points(c)
        ax.plot(x_m, y_m, "k--", linewidth=1.2, label=name)

    ax.set(xlabel="x1", ylabel="x2", title="Assignment 3: SVM training data",
           xlim=(-0.5, 2.5), ylim=(-0.5, 2.5))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def identify_support_vectors():
    """Q1(a), 10 points together with the plot: identify support vectors by eye.

    Return their 1-based point IDs and explain your choice in the report.
    """
    # TODO: Inspect the plot and identify all support vectors.
    return (1,5,6)


def derive_svm_separator():
    """Q1(b), 10 points for the boundary and 10 for the geometric margin.

    Use a linear, hard-margin SVM. The decision boundary is w dot x + b = 0;
    retain the intercept b. Use the support vectors to derive it.

    Return (weights, bias, geometric_margin), with weights a pair of numbers.
    Explain the derivation and your margin convention in the written report.
    """
    # TODO: Derive the maximum-margin separating line from the support vectors.
    # TODO: Determine the geometric margin and state its meaning.
    return ((2,2), -3, (2**0.5)/4)


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
