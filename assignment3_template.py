"""CSE 242 Assignment 3, Question 1: hard-margin linear SVM (30 points).
"""

# Outside-help acknowledgment: this solution was written with the help of
# Claude, a generative AI assistant made by Anthropic.

# Summary of the solution (the full derivation is in the written report):
#   * Support vectors (by eye, then verified): points 1 = (1, 1), 5 = (1, 0)
#     and 6 = (0, 1).
#   * Maximum-margin boundary: 2 x1 + 2 x2 - 3 = 0, i.e. w = (2, 2), b = -3
#     (equivalently x1 + x2 = 1.5).
#   * Geometric margin (boundary to the closest training points):
#     1 / ||w|| = 1 / sqrt(8) = sqrt(2) / 4 ~= 0.3536.
#     Full width between the supporting lines (reported separately):
#     2 / ||w|| = sqrt(2) / 2 ~= 0.7071.

import argparse
import math
from fractions import Fraction
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

    Answer: points 1 = (1, 1) [y = +1], 5 = (1, 0) [y = -1], 6 = (0, 1) [y = -1].

    Reasoning from the plot (svm_data.png):
    * No vertical or horizontal line separates the classes: negative point 5
      has the same x1 as positive point 1, and negative point 6 has the same
      x2. The boundary therefore has to be diagonal.
    * The side of the negative class that faces the positive class is the
      segment from point 5 to point 6, which lies on the line x1 + x2 = 1.
      The closest part of the positive class is the corner point 1, which lies
      on x1 + x2 = 2. The widest empty strip between the classes is
      1 <= x1 + x2 <= 2, and its two edges pass through exactly points 1, 5, 6.
    * Points 2 and 3 (x1 + x2 = 3) and point 4 (x1 + x2 = 0) lie behind that
      strip. Moving them slightly would not change the boundary, so they are
      not support vectors.

    derive_svm_separator() confirms this choice. Exactly these three points
    satisfy y_i (w . x_i + b) = 1, and all three get positive dual
    multipliers.
    """
    return [1, 5, 6]


def _solve_exact(matrix, rhs):
    """Solve a square linear system exactly (Gauss-Jordan with Fractions).

    Exact rational arithmetic keeps the derivation free of floating-point
    round-off, so w and b come out as exactly (2, 2) and -3.
    """
    size = len(matrix)
    rows = [[Fraction(value) for value in row] + [Fraction(target)]
            for row, target in zip(matrix, rhs)]
    for col in range(size):
        pivot = next((r for r in range(col, size) if rows[r][col] != 0), None)
        if pivot is None:
            raise ValueError("The support-vector equations are singular.")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        pivot_value = rows[col][col]
        rows[col] = [value / pivot_value for value in rows[col]]
        for r in range(size):
            if r != col and rows[r][col] != 0:
                factor = rows[r][col]
                rows[r] = [a - factor * p for a, p in zip(rows[r], rows[col])]
    return [row[-1] for row in rows]


def derive_svm_separator():
    """Q1(b), 10 points for the boundary and 10 for the geometric margin.

    Use a linear, hard-margin SVM. The decision boundary is w dot x + b = 0;
    retain the intercept b. Use the support vectors to derive it.

    Return (weights, bias, geometric_margin), with weights a pair of numbers.
    Explain the derivation and your margin convention in the written report.

    Derivation (the code below carries out these steps):
    1. Each support vector lies on its supporting line, y_i (w . x_i + b) = 1.
       Because y_i = +1 or -1, this is the same as w . x_i + b = y_i:
           point 1 (1, 1), y = +1:   w1 + w2 + b = +1
           point 5 (1, 0), y = -1:   w1      + b = -1
           point 6 (0, 1), y = -1:        w2 + b = -1
       Subtracting the second equation from the first gives w2 = 2.
       Subtracting the third from the first gives w1 = 2. Then b = -1 - w1 = -3.
       Boundary: 2 x1 + 2 x2 - 3 = 0, which is the line x1 + x2 = 1.5.
    2. Feasibility: y_i (w . x_i + b) equals 1 for points 1, 5 and 6 and
       equals 3 for points 2, 3 and 4. Every hard-margin constraint holds.
    3. Optimality (KKT): solve w = sum_i alpha_i y_i x_i together with
       sum_i alpha_i y_i = 0. This gives alpha_1 = 4 and alpha_5 = alpha_6 = 2,
       with alpha = 0 for points 2, 3 and 4. All multipliers are non-negative,
       so the KKT conditions of this convex problem hold. Hence (w, b) is the
       maximum-margin solution and not just some separating line.
    4. Geometric margin = 1 / ||w|| = 1 / sqrt(8) = sqrt(2) / 4 ~= 0.3536.
       This is the distance from the boundary to the closest training points
       (points 1, 5 and 6). The full width between the supporting lines
       x1 + x2 = 1 and x1 + x2 = 2 is 2 / ||w|| = sqrt(2) / 2 ~= 0.7071; that
       is a different quantity and is reported separately.
    """
    sv_ids = identify_support_vectors()
    sv_points = [SVM_X[i - 1] for i in sv_ids]
    sv_labels = [SVM_Y[i - 1] for i in sv_ids]

    # Step 1: one margin equation  w1 * x1 + w2 * x2 + b = y_i  per support vector.
    w1, w2, b = _solve_exact([[x1, x2, 1] for x1, x2 in sv_points], sv_labels)

    # Step 2: every point must satisfy y_i (w . x_i + b) >= 1, with equality
    # exactly for the chosen support vectors.
    functional_margins = [y * (w1 * x1 + w2 * x2 + b)
                          for (x1, x2), y in zip(SVM_X, SVM_Y)]
    if min(functional_margins) < 1:
        raise RuntimeError("The derived line violates a hard-margin constraint.")
    on_margin = [i for i, m in enumerate(functional_margins, start=1) if m == 1]
    if sorted(on_margin) != sorted(sv_ids):
        raise RuntimeError("The points on the margin do not match the chosen "
                           "support vectors.")

    # Step 3: KKT certificate. Solve sum_i alpha_i y_i x_i = w and
    # sum_i alpha_i y_i = 0 over the support vectors and require alpha_i > 0.
    kkt_matrix = [[y * x[0] for x, y in zip(sv_points, sv_labels)],
                  [y * x[1] for x, y in zip(sv_points, sv_labels)],
                  list(sv_labels)]
    alphas = _solve_exact(kkt_matrix, [w1, w2, 0])
    if any(alpha <= 0 for alpha in alphas):
        raise RuntimeError("A dual multiplier is not positive, so this is not "
                           "the maximum-margin solution.")

    # Step 4: geometric margin = distance from the boundary to the closest point.
    weight_norm = math.sqrt(float(w1 * w1 + w2 * w2))
    geometric_margin = 1.0 / weight_norm
    return (float(w1), float(w2)), float(b), geometric_margin


def format_boundary(weights, bias, multiply=" "):
    """Return the boundary as text, e.g. '2 x1 + 2 x2 - 3 = 0'."""
    w1, w2 = weights
    sign = "-" if bias < 0 else "+"
    return (f"{w1:g}{multiply}x1 + {w2:g}{multiply}x2 {sign} {abs(bias):g} = 0")


def plot_svm_solution(output_path):
    """Optional figure: the data with the derived boundary and margin.

    Draws the boundary w . x + b = 0, the supporting lines w . x + b = +/-1,
    circles the support vectors and marks the geometric margin. Everything is
    taken from identify_support_vectors() and derive_svm_separator().
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib could not be imported. Use the README's clean virtual "
            "environment and install requirements.txt."
        ) from exc

    (w1, w2), b, margin = derive_svm_separator()
    sv_ids = identify_support_vectors()

    fig, ax = plt.subplots(figsize=(6, 6))
    x_grid = [-0.5, 2.5]

    def line_x2(level):
        # Points on w1 * x1 + w2 * x2 + b = level, solved for x2 (w2 != 0 here).
        return [(level - b - w1 * x1) / w2 for x1 in x_grid]

    ax.fill_between(x_grid, line_x2(-1), line_x2(1), color="0.85", alpha=0.6,
                    label="margin band")
    sign = "-" if b < 0 else "+"
    boundary_tex = f"${w1:g}x_1 + {w2:g}x_2 {sign} {abs(b):g} = 0$"
    ax.plot(x_grid, line_x2(0), color="black", linewidth=2,
            label="boundary " + boundary_tex)
    ax.plot(x_grid, line_x2(1), color="tab:blue", linestyle="--", linewidth=1.2,
            label=r"$\mathbf{w}\cdot\mathbf{x} + b = +1$")
    ax.plot(x_grid, line_x2(-1), color="tab:orange", linestyle="--",
            linewidth=1.2, label=r"$\mathbf{w}\cdot\mathbf{x} + b = -1$")

    for label, color, marker in ((1, "tab:blue", "o"), (-1, "tab:orange", "s")):
        points = [x for x, y in zip(SVM_X, SVM_Y) if y == label]
        ax.scatter([x[0] for x in points], [x[1] for x in points], color=color,
                   marker=marker, s=90, zorder=3, label=f"y = {label:+d}")
    sv_points = [SVM_X[i - 1] for i in sv_ids]
    ax.scatter([x[0] for x in sv_points], [x[1] for x in sv_points], s=420,
               facecolors="none", edgecolors="crimson", linewidths=2, zorder=4,
               label="support vectors")
    for point_id, (x1, x2) in enumerate(SVM_X, start=1):
        ax.annotate(str(point_id), (x1, x2), xytext=(9, 7),
                    textcoords="offset points")

    # Arrow from support vector 1 to its orthogonal projection on the boundary.
    sx1, sx2 = SVM_X[sv_ids[0] - 1]
    norm_sq = w1 * w1 + w2 * w2
    step = (w1 * sx1 + w2 * sx2 + b) / norm_sq
    foot = (sx1 - step * w1, sx2 - step * w2)
    ax.annotate("", xy=foot, xytext=(sx1, sx2),
                arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.6))
    ax.text(foot[0] + 0.32, foot[1] + 0.02,
            r"margin $= 1/\Vert\mathbf{w}\Vert$" + f" = {margin:.4f}",
            color="crimson", fontsize=9)

    ax.set(xlabel="$x_1$", ylabel="$x_2$",
           title="Q1: maximum-margin boundary and support vectors",
           xlim=(-0.5, 2.5), ylim=(-0.5, 2.5))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plot", "svm", "boundary"),
                        default="plot",
                        help="Default: generate the SVM data plot. 'svm' prints "
                             "the derived solution; 'boundary' saves an optional "
                             "plot of the derived boundary and margin.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path. Defaults to svm_data.png for "
                             "--task plot and svm_boundary.png for --task "
                             "boundary, both beside the script.")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    if args.task == "plot":
        output = args.output or script_dir / "svm_data.png"
        print(f"Saved plot to {plot_svm_data(output)}")
    elif args.task == "boundary":
        output = args.output or script_dir / "svm_boundary.png"
        print(f"Saved boundary plot to {plot_svm_solution(output)}")
    else:
        print("Support vector point IDs:", identify_support_vectors())
        weights, bias, margin = derive_svm_separator()
        print(f"Weights: {weights}; bias: {bias}; geometric margin: {margin}")
        print(f"Decision boundary: {format_boundary(weights, bias, '*')}  "
              f"(equivalently x1 + x2 = {-bias / weights[0]:g})")
        print("Geometric margin = 1/||w|| = sqrt(2)/4 = "
              f"{margin:.6f} (distance from the boundary to points "
              f"{identify_support_vectors()})")
        print("Full width between supporting lines (2/||w||, reported "
              f"separately, not the margin): {2 * margin:.6f}")


if __name__ == "__main__":
    main()
