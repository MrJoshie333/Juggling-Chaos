import matplotlib.pyplot as plt
import numpy as np

# Data points
j_x = [0.4422, 0.5577, 0.4477]
j_y = [1.02425, 0.9424, 1.0477]

s_x = [5.6497, 2.7385, 0.3678]
s_y = [0.3542, 0.3976, 1.1185]

# Combine the data
x_all = j_x + s_x
y_all = j_y + s_y

# Create a figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot for j_x, j_y (red) and s_x, s_y (blue) on the first subplot
ax[0].scatter(j_x, j_y, color="red", label="J")
ax[0].scatter(s_x, s_y, color="blue", label="S")
ax[0].set_xlabel("l/f")
ax[0].set_ylabel("l/u")
ax[0].legend()
ax[0].set_title("Scatter Plot")

# Quadratic trendline for combined data (quadratic regression)
coefficients_all = np.polyfit(x_all, y_all, 2)  # degree 2 for quadratic fit
trendline_all = np.poly1d(coefficients_all)

# Second subplot with both scatter points and quadratic trendline
x_vals_all = np.linspace(min(x_all), max(x_all), 100)  # X values for the combined trendline

ax[1].scatter(j_x, j_y, color="red", label="J")
ax[1].scatter(s_x, s_y, color="blue", label="S")
ax[1].plot(x_vals_all, trendline_all(x_vals_all), color="green", label="Combined Quadratic Trendline")

ax[1].set_xlabel("l/f")
ax[1].set_ylabel("l/u")
ax[1].legend()
ax[1].set_title("Combined Quadratic Trendline")

# Display the plots
plt.tight_layout()
plt.show()

# Output the quadratic trendline equation for the combined data
print(f"Combined quadratic trendline equation: y = {coefficients_all[0]}x^2 + {coefficients_all[1]}x + {coefficients_all[2]}")
