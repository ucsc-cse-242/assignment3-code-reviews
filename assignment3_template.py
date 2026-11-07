"""CSE 242 Assignment 3, Question 1: hard-margin linear SVM (30 points).
"""

import argparse
from pathlib import Path


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


def identify_support_vectors():
    """Q1(a), 10 points together with the plot: identify support vectors by eye.

    Return their 1-based point IDs and explain your choice in the report.
    """
    # Positives (1,1),(1,2),(2,1) and negatives (0,0),(1,0),(0,1) are split by
    # lines of slope -1. The closest opposite-class pairs are (1,1) against
    # (1,0) and (0,1): each pair differs by one unit along an axis. (1,1) lies on
    # x1 + x2 = 2 and (1,0),(0,1) both lie on x1 + x2 = 1; every other point is
    # strictly farther out (x1 + x2 = 3 or 0). The widest street is the band
    # 1 <= x1 + x2 <= 2, whose two edges touch exactly points 1, 5 and 6.
    return [1, 5, 6]


def derive_svm_separator():
    """Q1(b), 10 points for the boundary and 10 for the geometric margin.

    Use a linear, hard-margin SVM. The decision boundary is w dot x + b = 0;
    retain the intercept b. Use the support vectors to derive it.

    Return (weights, bias, geometric_margin), with weights a pair of numbers.
    Explain the derivation and your margin convention in the written report.
    """
    # The support vectors sit on the edges of the street, so w . x + b = +1 at
    # (1,1) and w . x + b = -1 at (1,0) and (0,1).
    #   w1 + w2 + b = +1,   w1 + b = -1,   w2 + b = -1
    # The last two give w1 = w2; the first then gives w1 = w2 = 2, b = -3.
    # Boundary: 2 x1 + 2 x2 - 3 = 0, i.e. x1 + x2 = 1.5.
    # Geometric margin (boundary to the nearest training points) = 1 / ||w||
    #   = 1 / sqrt(8) = sqrt(2) / 4 ~ 0.3536. Full width 2 / ||w|| = sqrt(2) / 2.
    import math
    support = [SVM_X[i - 1] for i in identify_support_vectors()]
    labels = [SVM_Y[i - 1] for i in identify_support_vectors()]
    pos = [x for x, y in zip(support, labels) if y == 1][0]
    neg = [x for x, y in zip(support, labels) if y == -1]
    # Solve the 3x3 linear system w . x_s + b = y_s exactly (Cramer's rule).
    rows = [(pos[0], pos[1], 1, 1), (neg[0][0], neg[0][1], 1, -1),
            (neg[1][0], neg[1][1], 1, -1)]

    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    a = [r[:3] for r in rows]
    rhs = [r[3] for r in rows]
    base = det3(a)
    sol = []
    for col in range(3):
        m = [list(r) for r in a]
        for i in range(3):
            m[i][col] = rhs[i]
        sol.append(det3(m) / base)
    w1, w2, b = sol
    margin = 1 / math.hypot(w1, w2)
    return (w1, w2), b, margin


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
