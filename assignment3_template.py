"""CSE 242 Assignment 3, Question 1: hard-margin linear SVM (30 points).
"""

import argparse
import typing as ty
from pathlib import Path

import matplotlib.pyplot as plt

if ty.TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

SVM_X = ((1, 1), (1, 2), (2, 1), (0, 0), (1, 0), (0, 1))
SVM_Y = (1, 1, 1, -1, -1, -1)

def make_data_fig(title: str) -> tuple[Figure, Axes]:
    """
    Basic plotting of above SVM_ data.
    Returns the figure and axes object.
    It's the caller's job to figure out what to do with it.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    # this is not my code.
    # I did not write this.
    # I cannot be blamed for what it's doing here.
    # -Nicola
    for label, color, marker in ((1, "tab:blue", "o"), (-1, "tab:orange", "s")):
        points = [x for x, y in zip(SVM_X, SVM_Y) if y == label]
        ax.scatter(
            [x[0] for x in points], [x[1] for x in points],
            color=color, marker=marker, s=90, label=f"y = {label:+d}",
        )
    for point_id, (x1, x2) in enumerate(SVM_X, start=1):
        ax.annotate(str(point_id), (x1, x2), xytext=(7, 7),
                    textcoords="offset points")
    ax.set(
        title=title,
        xlabel="x1", ylabel="x2",
        xlim=(-0.5, 2.5), ylim=(-0.5, 2.5))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax

type SVM_X_1INDEX = int
EYEBALLED_SUPPORT_VECTORS: list[SVM_X_1INDEX] = [1, 5, 6]
EYEBALLED_MARGIN = (-.5, 2)
EYEBALLED_SLOPE = -1

def identify_support_vectors():
    """Q1(a), 10 points together with the plot: identify support vectors by eye.
    """

    # Return their 1-based point IDs
    return EYEBALLED_SUPPORT_VECTORS


def derive_svm_separator():
    """Q1(b), 10 points for the boundary and 10 for the geometric margin.

    Use a linear, hard-margin SVM. The decision boundary is w dot x + b = 0;
    retain the intercept b. Use the support vectors to derive it.

    Return (weights, bias, geometric_margin), with weights a pair of numbers.
    Explain the derivation and your margin convention in the written report.
    """
    return (1, 1), -1.5, 2**(-1.5)

def plot_svm_data(output: Path) -> None:
    """Generate an annotated scatter plot to help inspect Q1(a) by eye.

    Point IDs are 1-based and match the source table order. This helper
    does not identify support vectors or draw the separating boundary.
    """

    fig, _ = make_data_fig("Assignment 3: SVM training data")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

def plot_svm_data_with_support_vectors(output: Path) -> None:
    fig, ax = make_data_fig("Assignment 3: Hard Margin SVM")
    # Plot Support Vectors First
    for svm_x_1idx in EYEBALLED_SUPPORT_VECTORS:
        # gangster communist 1-indexing
        true_index = svm_x_1idx - 1
        ax.axline(SVM_X[true_index], slope=EYEBALLED_SLOPE,
                  color="black", dashes=(5, 5))

    # And then the margin
    ax.axline(EYEBALLED_MARGIN, slope=EYEBALLED_SLOPE, color="black")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

cwd = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("plot", "svm"), default="plot",
        help="Default: generate the SVM data plot.")
    parser.add_argument(
        "--output", type=Path, default=cwd / "svm_data.png",
        help="Output image path for --task plot.")
    args = parser.parse_args()
    output = args.output
    assert isinstance(output, Path)
    assert output.suffix == ".png", "Output image path must end with .png!"

    if args.task == "plot":
        plot_svm_data(output)
        print(f"Saved plot to {output}")
        output2 = output.with_stem(output.stem + "_svm")
        plot_svm_data_with_support_vectors(output2)
        print(f"Saved plot with support vectors to {output2}")
    else:
        print("Support vector point IDs:", identify_support_vectors())
        weights, bias, margin = derive_svm_separator()
        print(f"Weights: {weights}; bias: {bias}; geometric margin: {margin}")


if __name__ == "__main__":
    main()
