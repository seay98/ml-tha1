
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# A.I.1 - Data Acquisition (Group 17)
x = np.array([
    23.99, 26.53, 31.59, 24.06, 27.28,
    23.69, 11.46, 22.61, 21.84, 9.20,
    17.75, 7.53, 16.76, 22.69, 11.91,
    26.63, 6.72, 9.44, 18.32, 18.31
], dtype=float)

y = np.array([
    437.14, 439.96, 442.58, 436.88, 441.94,
    438.62, 470.55, 441.76, 450.88, 482.89,
    454.41, 479.15, 458.67, 446.91, 466.52,
    444.16, 489.09, 475.01, 471.43, 455.24
], dtype=float)


# A.I.2 - Data Transformation (Z-score Standardization)
def z_score(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std, mean, std


x_scaled, mu_x, sigma_x = z_score(x)
y_scaled, mu_y, sigma_y = z_score(y)


# A.I.3 - Least Squares
def least_squares(x_data, y_data):
    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    theta1 = np.sum((x_data - x_mean) * (y_data - y_mean)) / np.sum((x_data - x_mean) ** 2)
    theta0 = y_mean - theta1 * x_mean
    return theta0, theta1


def plot_least_squares(x_data, y_data, intercept, slope):
    x_line = np.linspace(x_data.min(), x_data.max(), 200)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(x_data, y_data, label="Group 17 observations")
    plt.plot(x_line, intercept + slope * x_line, label="Least-squares regression")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Net hourly electrical output (MW)")
    plt.title("A.I.3 — Least Squares (Group 17)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("a_i_3_least_squares.png", dpi=180)
    plt.close()

# A.I.4 - Gradient Descent: Cost Function
def cost_function(x, y, theta0, theta1):
    predictions = theta0 + theta1 * x
    errors = predictions - y
    return np.sum(errors ** 2) / (2 * len(x))


def compute_gradients(x, y, theta0, theta1):
    errors = theta0 + theta1 * x - y
    grad0 = np.mean(errors)
    grad1 = np.mean(errors * x)
    return grad0, grad1

# A.I.5 - Gradient Descent: First Iteration
def gradient_descent_step(x, y, theta0, theta1, learning_rate):
    grad0, grad1 = compute_gradients(x, y, theta0, theta1)

    # Update both parameters using the same gradients
    new_theta0 = theta0 - learning_rate * grad0
    new_theta1 = theta1 - learning_rate * grad1

    return new_theta0, new_theta1, grad0, grad1

# A.I.6 - Gradient Descent: Second Iteration

# A.I.7 - Gradient Descent: Last Iteration

# Plot cost
def plot_gd_cost_history(cost_history):
    plt.figure(figsize=(7, 4.5))

    plt.plot(range(len(cost_history)), cost_history)

    plt.xlabel("Gradient descent iteration")
    plt.ylabel("Cost J (standardized data)")
    plt.title("A.I.7 - Gradient Descent Convergence (Group 17)")

    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("a_i_7_cost_history.png", dpi=180)
    plt.close()

    print("Saved plot: a_i_7_cost_history.png")

# Plot function
def plot_gradient_descent(x_data, y_data, theta0, theta1,
                          mu_x, sigma_x, mu_y, sigma_y,
                          iteration):
    x_line = np.linspace(x_data.min(), x_data.max(), 200)

    # Standardize x
    x_line_scaled = (x_line - mu_x) / sigma_x

    # Predict and convert back to original units
    y_line_scaled = theta0 + theta1 * x_line_scaled
    y_line = y_line_scaled * sigma_y + mu_y

    plt.figure(figsize=(7, 4.5))
    plt.scatter(x_data, y_data, label="Group 17 observations")
    plt.plot(x_line, y_line,
             label=f"Gradient Descent - Iteration {iteration}")

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Net hourly electrical output (MW)")
    plt.title(f"Gradient Descent - Iteration {iteration}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    filename = f"a_i_iteration_{iteration}.png"
    plt.savefig(filename, dpi=180)
    plt.close()

    print(f"Saved plot: {filename}")

def main():
    x_scaled, mu_x, sigma_x = z_score(x)
    y_scaled, mu_y, sigma_y = z_score(y)

    # A.I.1
    print(f"A.I.1 - Group 17: {len(x)} data points")
    print("Temperature (x):", x)
    print("Power output (y):", y)

    # A.I.2
    print("\nA.I.2 - Z-score Standardization")
    print(f"mu_x = {mu_x:.4f}, sigma_x = {sigma_x:.4f}")
    print(f"mu_y = {mu_y:.4f}, sigma_y = {sigma_y:.4f}")
    print("x_scaled:", np.round(x_scaled, 4))
    print("y_scaled:", np.round(y_scaled, 4))

    # A.I.3
    theta0, theta1 = least_squares(x_scaled, y_scaled)
    slope = theta1 * sigma_y / sigma_x
    intercept = mu_y + sigma_y * theta0 - slope * mu_x
    print("\nA.I.3 - Least Squares")
    print(f"Standardized parameters: theta0={theta0:.6f}, theta1={theta1:.6f}")
    print(f"Original units: intercept={intercept:.6f} MW, slope={slope:.6f} MW/°C")
    plot_least_squares(x, y, intercept, slope)
    print("Saved plot: a_i_3_least_squares.png")

    # A.I.4 - Gradient Descent: Cost Function

    # A.I.5 - Gradient Descent: First Iteration
    rng = np.random.default_rng(42)
    gd_theta0, gd_theta1 = rng.uniform(-1, 1, size=2)
    learning_rate = 0.1

    print("\nA.I.5 - Gradient Descent: First Iteration")
    print(f"Learning rate: {learning_rate}")
    print(f"Initial parameters: theta0={gd_theta0:.6f}, "
          f"theta1={gd_theta1:.6f}")

    initial_cost = cost_function(
        x_scaled, y_scaled, gd_theta0, gd_theta1
    )

    gd_theta0, gd_theta1, grad0, grad1 = gradient_descent_step(
        x_scaled, y_scaled, gd_theta0, gd_theta1, learning_rate
    )

    first_cost = cost_function(
        x_scaled, y_scaled, gd_theta0, gd_theta1
    )

    print(f"Gradients: grad0={grad0:.6f}, grad1={grad1:.6f}")
    print(f"Updated parameters: theta0={gd_theta0:.6f}, "
          f"theta1={gd_theta1:.6f}")
    print(f"Initial cost: {initial_cost:.6f}")
    print(f"Cost after iteration 1: {first_cost:.6f}")
    plot_gradient_descent(
        x, y, gd_theta0, gd_theta1,
        mu_x, sigma_x, mu_y, sigma_y,
        iteration=1
    )

    # A.I.6 - Gradient Descent: Second Iteration
    gd_theta0, gd_theta1, grad0, grad1 = gradient_descent_step(
    x_scaled, y_scaled, gd_theta0, gd_theta1, learning_rate
    )

    second_cost = cost_function(
        x_scaled, y_scaled, gd_theta0, gd_theta1
    )

    print("\nA.I.6 - Gradient Descent: Second Iteration")
    print(f"Gradients: grad0={grad0:.6f}, grad1={grad1:.6f}")
    print(f"Updated parameters: theta0={gd_theta0:.6f}, "
        f"theta1={gd_theta1:.6f}")
    print(f"Cost after iteration 2: {second_cost:.6f}")
    plot_gradient_descent(
        x, y, gd_theta0, gd_theta1,
        mu_x, sigma_x, mu_y, sigma_y,
        iteration=2
    )

    # A.I.7 - Gradient Descent: Last Iteration
    cost_history = [initial_cost, first_cost, second_cost]
    max_iterations = 1000
    tolerance = 1e-10
    final_iteration = 2

    for iteration in range(3, max_iterations + 1):
        gd_theta0, gd_theta1, _, _ = gradient_descent_step(
            x_scaled, y_scaled, gd_theta0, gd_theta1, learning_rate
        )

        current_cost = cost_function(
            x_scaled, y_scaled, gd_theta0, gd_theta1
        )

        cost_history.append(current_cost)
        final_iteration = iteration

        if abs(cost_history[-2] - current_cost) < tolerance:
            break

    # Convert parameters back to original units
    gd_slope = gd_theta1 * sigma_y / sigma_x
    gd_intercept = mu_y + sigma_y * gd_theta0 - gd_slope * mu_x

    print("\nA.I.7 - Gradient Descent: Last Iteration")
    print(f"Iterations completed: {final_iteration}")
    print(f"Final standardized parameters: theta0={gd_theta0:.6f}, theta1={gd_theta1:.6f}")
    print(f"Final original units: intercept={gd_intercept:.6f} MW, slope={gd_slope:.6f} MW/°C")
    print(f"Final cost: {cost_history[-1]:.8f}")
    print(f"Least-squares cost: {cost_function(x_scaled, y_scaled, theta0, theta1):.8f}")

    # Plot final regression line
    plot_gradient_descent(
        x, y, gd_theta0, gd_theta1,
        mu_x, sigma_x, mu_y, sigma_y,
        iteration=final_iteration
    )

    # Plot cost convergence
    plot_gd_cost_history(cost_history)


if __name__ == "__main__":
    main()