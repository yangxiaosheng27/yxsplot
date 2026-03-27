import numpy as np
from yxsplot import plot, show_figure

# Generate 1 million data points
n = 1_000_000
x = np.linspace(0, 2 * np.pi, n)
y1 = np.sin(x)
y2 = np.cos(x)

# --- Plot 1: Base Sine Wave (Master View) ---
# Creates the first figure window and axes. This serves as the "master" 
# for sharing coordinates with other plots.
ax1 = plot(
    x,
    y1,
    data_name="Sine",
    title="Overlay & Linking Demo (Master View)",
)

# --- Plot 2: Overlay Curve on Same Axes (Not a Subplot) ---
# Setting 'new_fig=False' draws this curve on the **exact same** axes object as ax1.
# This adds a new curve that shares the identical X/Y axis scales and limits 
# with the previous one, displaying both lines overlaid on the same canvas.
ax2 = plot(
    x, 
    y2, 
    data_name="Cosine", 
    new_fig=False,  # Critical: Do not create a new window/axes; overlay directly onto ax1
)

# --- Plot 3: Fully Linked View (Synchronized Zoom/Pan) ---
# This creates a **new**, independent figure window, but binds its axes to ax1.
# When you zoom or pan in 'ax1' (the first window), 'ax3' (this window) updates identically.
ax3 = plot(
    x,
    y1,
    data_name="Sine Color Map",
    share_x=ax1,      # Link X-axis: Zooming/panning ax1 automatically syncs ax3's X view
    share_y=ax1,      # Link Y-axis: Zooming/panning ax1 automatically syncs ax3's Y view
    title="Linked View",
    color=y1,         # Enable color mapping: Display gradient based on Y values
    color_name="Value",
    # Note: Since 'new_fig=False' is not set here, this opens in a separate window.
    # However, it stays perfectly synchronized with the first window. 
    # Ideal for comparing views (e.g., raw waveforms vs. heatmaps).
)

# Render and display all created interactive windows.
# Result: Two windows will pop up:
# 1. [Overlay Plot]: Sine and Cosine curves displayed together in the same coordinate system.
# 2. [Linked Plot]: An independent color-mapped plot whose view range syncs in real-time with Window 1.
show_figure()