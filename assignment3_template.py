"""CSE 242 Assignment 3, Question 1: hard-margin linear SVM (30 points).

Student: Kangan Qian (SID 2346765)

Solution summary (full derivation in the written report):
- Support vectors: points 1 (1,1), 5 (1,0), 6 (0,1). They are the only points
  at distance sqrt(2)/4 from the optimal boundary; every other point is at
  distance 3*sqrt(2)/4, strictly greater.
- Optimal separator (canonical scaling): w = (2, 2), b = -3, i.e. the line
  2*x1 + 2*x2 - 3 = 0, or x1 + x2 = 1.5.
- Geometric margin (boundary -> closest points): 1/||w|| = sqrt(2)/4 ≈ 0.3536.
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
    # Reading the plot, the two classes are separated with the largest gap
    # along the diagonal direction x1 + x2. The margin strip of the optimal
    # boundary x1 + x2 = 1.5 runs between the lines x1 + x2 = 1 and
    # x1 + x2 = 2. Exactly three points touch that strip:
    #   point 1 (1, 1): x1 + x2 = 2 (closest positive point to the boundary),
    #   point 5 (1, 0): x1 + x2 = 1 (closest negative point),
    #   point 6 (0, 1): x1 + x2 = 1 (closest negative point).
    # Points 2 (1,2) and 3 (2,1) have x1 + x2 = 3 and point 4 (0,0) has
    # x1 + x2 = 0: all three sit strictly outside the margin strip, so none
    # of them can pin down the boundary (removing them would not move the
    # optimum). Hence the support vectors are points 1, 5 and 6.
    return [1, 5, 6]


def derive_svm_separator():
    """Q1(b), 10 points for the boundary and 10 for the geometric margin.

    Use a linear, hard-margin SVM. The decision boundary is w dot x + b = 0;
    retain the intercept b. Use the support vectors to derive it.

    Return (weights, bias, geometric_margin), with weights a pair of numbers.
    Explain the derivation and your margin convention in the written report.
    """
    import numpy as np

    # By symmetry of the point set about the line x1 = x2 (swapping the two
    # coordinates maps positives to positives and negatives to negatives),
    # the maximum-margin boundary must be invariant under that swap, so it
    # has the form w1 = w2. The remaining unknown is the scale of w and the
    # intercept; fix the scale with the SVM convention that the support
    # vectors lie exactly on the supporting hyperplanes y (w . x + b) = 1.
    # That gives three linear equations in (w1, w2, b):
    #   point 1 (1, 1), y = +1:  w1 + w2 + b =  1
    #   point 5 (1, 0), y = -1:   w1      + b = -1
    #   point 6 (0, 1), y = -1:        w2 + b = -1
    equations = np.array(
        [[1.0, 1.0, 1.0],   # coefficients of (w1, w2, b) for point 1
         [1.0, 0.0, 1.0],   # coefficients of (w1, w2, b) for point 5
         [0.0, 1.0, 1.0]]   # coefficients of (w1, w2, b) for point 6
    )
    targets = np.array([1.0, -1.0, -1.0])
    w1, w2, b = np.linalg.solve(equations, targets)
    weights = (float(w1), float(w2))

    # Sanity check: every training point must satisfy the hard-margin
    # constraint y (w . x + b) >= 1. Points on the wrong side or inside the
    # margin would mean the support-vector choice was wrong.
    margins = [y * (weights[0] * x1 + weights[1] * x2 + b)
               for (x1, x2), y in zip(SVM_X, SVM_Y)]
    if min(margins) < 1.0 - 1e-9:
        raise ValueError("Candidate separator violates a hard-margin constraint")

    # Geometric margin: the distance from the boundary w . x + b = 0 to the
    # closest training points (the support vectors, where the functional
    # margin equals 1) is 1 / ||w||. With w = (2, 2) this is 1 / (2*sqrt(2)).
    geometric_margin = 1.0 / float(np.hypot(w1, w2))
    return weights, float(b), geometric_margin


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
