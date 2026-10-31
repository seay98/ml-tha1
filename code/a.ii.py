from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Group 17: Excel rows 322-341 (temperature in Celsius, power in MW).
DATA = np.array([
    [23.99, 437.14], [26.53, 439.96], [31.59, 442.58], [24.06, 436.88],
    [27.28, 441.94], [23.69, 438.62], [11.46, 470.55], [22.61, 441.76],
    [21.84, 450.88], [9.20, 482.89], [17.75, 454.41], [7.53, 479.15],
    [16.76, 458.67], [22.69, 446.91], [11.91, 466.52], [26.63, 444.16],
    [6.72, 489.09], [9.44, 475.01], [18.32, 471.43], [18.31, 455.24],
], dtype=float)


def z_score(values):
    """Standardize values: z = (value - mean) / standard_deviation."""
    mean = np.mean(values)
    std = np.std(values, ddof=0)  # Population standard deviation.
    if std == 0:
        raise ValueError("Cannot standardize a constant array")
    return (values - mean) / std, mean, std


def inverse_z_score(z, mean, std):
    """Convert standardized predictions back to original units."""
    return z * std + mean


def model(x, theta):
    """Quadratic model: h(x) = theta[0] + theta[1]*x + theta[2]*x^2."""
    return theta[0] + theta[1] * x + theta[2] * x**2


def cost(x, y, theta):
    """Assignment's quartic cost: J = (1/(4*n)) * sum((y - h(x))^4)."""
    error = y - model(x, theta)
    return np.mean(error**4) / 4.0


def gradient(x, y, theta):
    """Return dJ/dtheta0, dJ/dtheta1, dJ/dtheta2 (derived in A.II.1)."""
    error_cubed = (y - model(x, theta))**3
    return -np.array([
        np.mean(error_cubed),
        np.mean(error_cubed * x),
        np.mean(error_cubed * x**2),
    ])


def gradient_descent(x, y, initial_theta, learning_rate, max_iterations,
                     tolerance):
    """Batch gradient descent; record the loss and selected parameter snapshots.

    Each gradient is computed from the same old theta before updating all
    three parameters simultaneously. Stop when the gradient norm is small.
    """    
    theta = np.array(initial_theta, dtype=float).copy()
    loss_history = [cost(x, y, theta)]  # Entry 0 is before any update.
    snapshots = {0: theta.copy()}
    snapshot_steps = {1, 10, 100, 1000}

    for step in range(1, max_iterations + 1):
        grad = gradient(x, y, theta)
        if np.linalg.norm(grad) < tolerance:
            break

        theta = theta - learning_rate * grad
        if not np.all(np.isfinite(theta)):
            raise ValueError("Gradient descent diverged; try to lower learning_rate")

        loss_history.append(cost(x, y, theta))
        if step in snapshot_steps:
            snapshots[step] = theta.copy()

    updates = len(loss_history) - 1
    snapshots[updates] = theta.copy()
    return theta, np.array(loss_history), snapshots


def plot_fitting_stages(x, y, snapshots):
    """Show how the regression curve changes during training (Z-score units)."""
    stages = sorted(snapshots)
    x_line = np.linspace(x.min() - 0.15, x.max() + 0.15, 300)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)

    for ax, step in zip(axes.flat, stages):
        ax.scatter(x, y, s=22, label="Data")
        ax.plot(x_line, model(x_line, snapshots[step]),
                color="tab:orange", label="Model")
        ax.set_title(f"Iteration {step}")
        ax.grid(alpha=0.2)

    for ax in axes[-1]:
        ax.set_xlabel("Standardized temperature")
    for ax in axes[:, 0]:
        ax.set_ylabel("Standardized power")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig("a.ii.fitting_stages.png", dpi=180)
    plt.close(fig)


def plot_loss(loss_history):
    """Plot the initial loss and the first 100 updates on a linear scale."""
    early_count = min(101, len(loss_history))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(early_count), loss_history[:early_count])
    ax.set(xlabel="Completed updates", ylabel="Quartic cost J (standardized y)",
           title="Gradient descent convergence (first 100 updates)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("a.ii.loss_curve.png", dpi=180)
    plt.close(fig)


def plot_final_fit(x_original, y_original, theta, x_mean, x_std,
                   y_mean, y_std):
    """Plot final fitted curve in Celsius and MW (not standardized units)."""
    x_line = np.linspace(x_original.min(), x_original.max(), 300)
    x_line_z = (x_line - x_mean) / x_std
    y_line = inverse_z_score(model(x_line_z, theta), y_mean, y_std)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x_original, y_original, label="Observed data")
    ax.plot(x_line, y_line, color="tab:orange", label="Polynomial model")
    ax.set(xlabel="Temperature (°C)", ylabel="Power output (MW)",
           title="Polynomial regression — Group 17")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("a.ii.final_fit_original_units.png", dpi=180)
    plt.close(fig)


def main():
    # 1. Load the fixed Group 17 arrays (no Excel reading required).
    x_original = DATA[:, 0]
    y_original = DATA[:, 1]

    # 2. Standardize both input and output using Z-score.
    x, x_mean, x_std = z_score(x_original)
    y, y_mean, y_std = z_score(y_original)

    # 3. Set gradient descent hyperparameters.
    # Random initialization for model parameters
    rng = np.random.default_rng(seed=42)
    initial_theta = rng.uniform(-1.0, 1.0, size=3)
    learning_rate = 0.02
    max_iterations = 10000
    tolerance = 1e-10

    # 4. Train the quadratic polynomial model.
    theta, loss_history, snapshots = gradient_descent(
        x, y, initial_theta, learning_rate, max_iterations, tolerance
    )

    # 5. Evaluate the fitted model in the original MW units.
    y_pred_z = model(x, theta)
    y_pred_mw = inverse_z_score(y_pred_z, y_mean, y_std)
    # rmse_mw = np.sqrt(np.mean((y_original - y_pred_mw)**2))

    print("Group 17 | 20 data points | Excel rows 322-341")
    print(f"Initial theta: {initial_theta}")
    print(f"Learning rate: {learning_rate}")
    print(f"Completed updates: {len(loss_history) - 1}")
    print(f"Final theta: {theta}")
    print(f"Initial cost: {loss_history[0]:.9f}")
    print(f"Final cost: {loss_history[-1]:.9f}")
    # print(f"Training RMSE: {rmse_mw:.4f} MW")

    # 6. Create three figures in the same folder as this Python file.
    output_dir = Path(__file__).resolve().parent
    plot_fitting_stages(x, y, snapshots)
    plot_loss(loss_history)
    plot_final_fit(x_original, y_original, theta, x_mean, x_std,
                   y_mean, y_std)
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
