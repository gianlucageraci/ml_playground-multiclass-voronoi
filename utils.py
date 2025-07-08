import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from net import Net


def get_decision_boundaries(weights, biases):
    """Compute decision boundary parameters (w_i - w_j, b_i - b_j)
    for each pair of classes from the final layer.
    """
    return [
        [w1 - w2, b1 - b2]
        for i, (w1, b1) in enumerate(zip(weights, biases))
        for w2, b2 in zip(weights[i + 1 :], biases[i + 1 :])
    ]


def plot_voronoi_from_logits(
    net: Net,
    epoch: int,
    loss: torch.Tensor,
    with_decision_boundary: bool = False,
    xlim=(-5, 5),
    ylim=(-5, 5),
    resolution=300,
):
    """Plot 2D Voronoi partitioning, decision boundaries, and weight vectors of the final layer."""
    weights = net.linear_stack[1].weight.detach().numpy()
    biases = net.linear_stack[1].bias.detach().numpy()
    num_classes = weights.shape[0]

    out_dir = f"figures/{num_classes}_classes"
    if with_decision_boundary:
        out_dir = f"figures/{num_classes}_classes/w_db"
    os.makedirs(out_dir, exist_ok=True)

    # Create grid
    xx, yy = np.meshgrid(np.linspace(*xlim, resolution), np.linspace(*ylim, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict class for each grid point
    logits = grid @ weights.T + biases
    zz = np.argmax(logits, axis=1).reshape(resolution, resolution)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot filled contours
    cmap = plt.get_cmap("tab10", num_classes)
    contour = ax.contourf(
        xx, yy, zz, levels=np.arange(num_classes + 1) - 0.5, cmap=cmap, alpha=0.75
    )

    # Add legend entries for each class
    for class_idx in range(num_classes):
        ax.plot(
            [],
            [],
            color=cmap(class_idx),
            marker="s",
            linestyle="None",
            markersize=10,
            label=f"Class {class_idx}",
            alpha=0.75,
        )

    # Plot weight vectors as arrows from origin
    for i, w in enumerate(weights):
        ax.arrow(
            0,
            0,
            *w,
            color="black",
            width=0.02,
            head_width=0.1,
            alpha=0.6,
            length_includes_head=True,
        )
        ax.text(
            w[0] * 1.1,
            w[1] * 1.1,
            f"$w_{i}$",
            fontsize=8,
            ha="center",
            va="center",
            weight="bold",
        )

    if with_decision_boundary:
        # Plot decision boundaries (dashed lines)
        label_added = False
        for w_diff, b_diff in get_decision_boundaries(weights, biases):
            if np.allclose(w_diff, 0):
                continue
            if w_diff[0] == 0:
                x = np.linspace(*xlim, 1000)
                y = np.full_like(x, -b_diff / w_diff[1])
            elif w_diff[1] == 0:
                y = np.linspace(*ylim, 1000)
                x = np.full_like(y, -b_diff / w_diff[0])
            else:
                x = np.linspace(*xlim, 1000)
                y = (-b_diff - w_diff[0] * x) / w_diff[1]
                mask = (y >= ylim[0]) & (y <= ylim[1])
                x, y = x[mask], y[mask]

            label = "Decision Boundaries" if not label_added else None
            ax.plot(x, y, "k--", linewidth=1, alpha=0.35, label=label)
            label_added = True

    # Set limits and aspect ratio
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(
        f"Multiclass Partitioning with {num_classes} Classes - Epoch {epoch} - Loss {loss.item():.2f}",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    plt.savefig(f"{out_dir}/voronoi_epoch_{epoch}.png", dpi=450, bbox_inches="tight")
    plt.close(fig)  # Close figure to avoid memory issues in loops
