import numpy as np
from yxsplot import plot, show_figure

# Generate 1 million points for a unit circle
n = 1_000_000
t = np.linspace(0, 2 * np.pi, n)
x = np.cos(t)
y = np.sin(t)

# Prepare custom metadata to display in interactive data info.
# Keys become labels, values must be arrays of length n.
data_info = {
    "Category": ["Circle"] * n,      # Constant string for all points
    "Tag": ["OK"] * n,               # Custom tag string
    "Flag (y > 0)": y > 0,                   # Boolean mask (e.g., Upper half)
}

# --- Plot 1: Logarithmic Scale & Custom Marker ---
# Transforms coordinates using exp() to demonstrate log-scale visualization.
# Since x and y are from a circle (-1 to 1), exp(x) and exp(y) will be positive.
ax = plot(
    np.exp(x),                # X data (transformed)
    np.exp(y),                # Y data (transformed)
    data_name="exp circle",   # Legend label
    x_name="log x",           # X-axis label
    y_name="log y",           # Y-axis label
    log_x=True,               # Enable logarithmic scale for X-axis
    log_y=True,               # Enable logarithmic scale for Y-axis
    line_marker="*",          # Use star markers instead of default lines/dots
    title="Advanced: Log Scale & Custom Line Marker",
)

# --- Plot 2: Equal Aspect Ratio, Color Mapping & Data Info ---
# Plots the original circle with additional interactive features.
ax2 = plot(
    x,                        # Original X data
    y,                        # Original Y data
    x_name="x",               # X-axis label
    y_name="y",               # Y-axis label
    color=t,                  # Map the parameter 't' (angle) to color
    color_name="t",           # Label for the color bar
    data_info=data_info,      # Attach custom tooltip data defined above
    equal_scale=True,         # Force 1:1 aspect ratio (crucial for circles)
    time_range=[1, 5],        # Initialize the time-range slider with this window
    title="Advanced: Equal Scale & Custom Data Info",
)

# Render and display the interactive window containing both plots
show_figure()