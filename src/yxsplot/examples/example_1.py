import numpy as np
from yxsplot import plot, show_figure

# Generate 1 million points
n = 1_000_000
x = np.linspace(0, 2 * np.pi, n)
y = np.sin(x)

# Create an interactive plot
# 'mask' allows you to filter data visually without changing the source array
ax = plot(
    y,
    data_name="Sine Wave (y > -0.5)",
    y_name="Amplitude",
    title="Basic Plotting Demo (1M Points)",
    mask=y > -0.5,  # Only display points where y > -0.5
)

# Show the plot window
show_figure()