from yxsplot import plot, show_figure, welch
import numpy as np

# Sampling parameters
Ts = 1e-3              # Sampling period (s)
fs = 1 / Ts            # Sampling frequency (Hz)
t = np.arange(1000) * Ts  # Time vector (1 second duration)

# Generate test signal: two sine waves (20.5 Hz and 120 Hz) + DC offset
u = 1 * np.sin(20.5 * 2 * np.pi * t) + 1 * np.sin(120.0 * 2 * np.pi * t) + 1

# Case 1: Segmented Welch with Hann window (50% overlap by default)
# Reduces variance through averaging but broadens peaks due to shorter segments
freq1, amp1, phase1 = welch(
    u, mode="amplitude", fs=fs, window="hann", nperseg=len(u) // 2
)

# Case 2: Single segment with Hann window (no segmentation, no averaging)
# Better frequency resolution but higher variance; reduced spectral leakage
freq2, amp2, phase2 = welch(u, mode="amplitude", fs=fs, window="hann")

# Case 3: Single segment with rectangular window (no window, no segmentation)
# Narrowest peaks (best resolution) but significant spectral leakage (sidelobes)
freq3, amp3, phase3 = welch(u, mode="amplitude", fs=fs, window=None)

# Plot comparison of three windowing/segmentation approaches
plot(
    freq1,
    amp1,
    data_name="Segmented + Hann Window",
    x_name="Frequency (Hz)",
    y_name="Amplitude",
    title="Spectral Analysis Comparison",
)

plot(
    freq2,
    amp2,
    data_name="No Segmentation + Hann Window",
    x_name="Frequency (Hz)",
    y_name="Amplitude",
    title="Spectral Analysis Comparison",
    new_fig=False,
)

plot(
    freq3,
    amp3,
    data_name="No Segmentation + Rectangular Window",
    x_name="Frequency (Hz)",
    y_name="Amplitude",
    title="Spectral Analysis Comparison",
    new_fig=False,
)

show_figure()