
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


# A.I.5 - Gradient Descent: First Iteration


# A.I.6 - Gradient Descent: Second Iteration


# A.I.7 - Gradient Descent: Last Iteration


# A.I.8 - Discussion

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
    return x_scaled, y_scaled, mu_x, sigma_x, mu_y, sigma_y


    # A.I.4 - Gradient Descent: Cost Function

    # A.I.5 - Gradient Descent: First Iteration

    # A.I.6 - Gradient Descent: Second Iteration

    # A.I.7 - Gradient Descent: Last Iteration

    # A.I.8 - Discussion


if __name__ == "__main__":
    main()