from typing import Dict, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    losses: Dict[str, List[float]], save_path: str = "training_curves.png"
):
    """
    Plot training loss curves with separate subplots for energy, forces, and stress.

    Args:
        losses: Dictionary containing lists of losses
            {'energy_loss': [...], 'force_loss': [...], 'stress_loss': [...]}
        save_path: Path to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    steps = np.arange(len(losses["energy_loss"]))

    # Energy loss subplot
    ax1.plot(steps, losses["energy_loss"], "b-", linewidth=2, label="Energy Loss")
    ax1.set_ylabel("Energy Loss (eV²)")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()

    # Force loss subplot
    ax2.plot(steps, losses["force_loss"], "g-", linewidth=2, label="Force Loss")
    ax2.set_ylabel("Force Loss (eV²/Å²)")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()

    # Stress loss subplot
    ax3.plot(steps, losses["stress_loss"], "r-", linewidth=2, label="Stress Loss")
    ax3.set_xlabel("Training Steps")
    ax3.set_ylabel("Stress Loss (GPa²)")
    ax3.set_yscale("log")
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    ax3.legend()

    # Add overall title
    plt.suptitle("Training Loss Curves", fontsize=14, y=0.95)

    # Add moving averages
    window = max(len(steps) // 50, 1)  # Dynamic window size
    for ax, loss_key in zip(
        [ax1, ax2, ax3], ["energy_loss", "force_loss", "stress_loss"]
    ):
        moving_avg = np.convolve(
            losses[loss_key], np.ones(window) / window, mode="valid"
        )
        ax.plot(
            steps[window - 1 :],
            moving_avg,
            "k--",
            alpha=0.7,
            linewidth=1.5,
            label=f"Moving Avg (window={window})",
        )

        # Add min value annotation
        min_loss = min(losses[loss_key])
        ax.axhline(y=min_loss, color="gray", linestyle=":", alpha=0.5)
        ax.text(
            0.02,
            0.98,
            f"Min Loss: {min_loss:.2e}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_parity_plots(
    predictions: Dict[str, jax.Array],
    labels: Dict[str, jax.Array],
    save_path: str = "parity_plots.png",
):
    """
    Create parity plots for energy, forces, and stress with improved styling.

    Args:
        predictions: Dictionary containing model predictions
            {'energy': [...], 'forces': [...], 'stress': [...]}
        labels: Dictionary containing true labels
            {'energy': [...], 'forces': [...], 'stress': [...]}
        save_path: Path to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Common plotting parameters
    scatter_params = dict(alpha=0.5, s=20, rasterized=True)
    line_params = dict(color="r", linestyle="--", label="y=x")

    # Energy parity plot
    energy_pred = predictions["energy"].ravel()
    energy_true = labels["energy"].ravel()
    plot_parity(
        ax1, energy_true, energy_pred, "Energy (eV)", scatter_params, line_params
    )

    # Forces parity plot
    forces_pred = predictions["forces"].ravel()
    forces_true = labels["forces"].ravel()
    plot_parity(
        ax2, forces_true, forces_pred, "Forces (eV/Å)", scatter_params, line_params
    )

    # Stress parity plot
    stress_pred = predictions["stress"].ravel()
    stress_true = labels["stress"].ravel()
    plot_parity(
        ax3, stress_true, stress_pred, "Stress (eV/Å³)", scatter_params, line_params
    )

    plt.suptitle("Parity Plots: Predicted vs True Values", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_parity(ax, true_vals, pred_vals, label, scatter_params, line_params):
    """Helper function for creating individual parity plots."""
    ax.scatter(true_vals, pred_vals, **scatter_params)

    # Get axis limits
    min_val = min(jnp.min(true_vals), jnp.min(pred_vals))
    max_val = max(jnp.max(true_vals), jnp.max(pred_vals))
    buffer = (max_val - min_val) * 0.1
    ax.plot(
        [min_val - buffer, max_val + buffer],
        [min_val - buffer, max_val + buffer],
        **line_params,
    )

    # Calculate and display metrics
    rmse = jnp.sqrt(jnp.mean((pred_vals - true_vals) ** 2))
    mae = jnp.mean(jnp.abs(pred_vals - true_vals))
    r2 = 1 - jnp.sum((pred_vals - true_vals) ** 2) / jnp.sum(
        (true_vals - jnp.mean(true_vals)) ** 2
    )

    stats_text = f"RMSE: {rmse:.2e}\n" f"MAE: {mae:.2e}\n" f"R²: {r2:.3f}"

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")


# Example usage remains the same as before:
def plot_training_results(
    training_losses: Dict[str, List[float]],
    final_predictions: Dict[str, jax.Array],
    final_labels: Dict[str, jax.Array],
    step_count: int,
):
    """
    Create and save both training curves and parity plots.
    """
    plot_training_curves(training_losses, f"training_curves_{step_count}.png")
    create_parity_plots(
        final_predictions, final_labels, f"parity_plots_{step_count}.png"
    )
