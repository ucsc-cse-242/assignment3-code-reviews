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
    # The closest opposite-class points are point 1 (1, 1) on the positive
    # side and points 5 (1, 0) and 6 (0, 1) on the negative side. Points 5 and
    # 6 lie on the line x1 + x2 = 1, point 1 lies on x1 + x2 = 2, and the
    # remaining points (2, 3, 4) are farther from the gap between the classes.
    return [1, 5, 6]


def derive_svm_separator():
    """Q1(b), 10 points for the boundary and 10 for the geometric margin.

    Use a linear, hard-margin SVM. The decision boundary is w dot x + b = 0;
    retain the intercept b. Use the support vectors to derive it.

    Return (weights, bias, geometric_margin), with weights a pair of numbers.
    Explain the derivation and your margin convention in the written report.
    """
    import math

    # Support vectors satisfy y_i (w . x_i + b) = 1:
    #   point 1 (1, 1), y = +1:   w1 + w2 + b =  1
    #   point 5 (1, 0), y = -1:  -(w1 + b)    =  1  ->  w1 + b = -1
    #   point 6 (0, 1), y = -1:  -(w2 + b)    =  1  ->  w2 + b = -1
    # Subtracting the third equation from the first gives w1 = 2, and
    # subtracting the second from the first gives w2 = 2; then b = -1 - w1 = -3.
    # The same system is built from the support vectors and solved below
    # (Cramer's rule), so the result comes from the data rather than hard-coding.
    rows = []
    for point_id in identify_support_vectors():
        (x1, x2), y = SVM_X[point_id - 1], SVM_Y[point_id - 1]
        # y (w1 x1 + w2 x2 + b) = 1  ->  [y x1, y x2, y] . [w1, w2, b] = 1
        rows.append((y * x1, y * x2, y, 1))

    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    coeffs = [row[:3] for row in rows]
    rhs = [row[3] for row in rows]
    denom = det3(coeffs)
    unknowns = []
    for col in range(3):
        replaced = [[rhs[r] if c == col else coeffs[r][c] for c in range(3)]
                    for r in range(3)]
        unknowns.append(det3(replaced) / denom)
    weights = (unknowns[0], unknowns[1])  # (2.0, 2.0)
    bias = unknowns[2]                    # -3.0

    # Geometric margin = distance from the boundary to the closest training
    # points = 1 / ||w||. (The full width between the two supporting lines,
    # 2 / ||w||, is reported separately in the write-up.)
    geometric_margin = 1.0 / math.sqrt(weights[0] ** 2 + weights[1] ** 2)
    return weights, bias, geometric_margin


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
