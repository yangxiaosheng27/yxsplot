# yxsplot

**Smooth and interactive 2D plotting for large-scale datasets.**

🌍 **Language: [English](#english) | [中文](#中文)**

---

## English

`yxsplot` is a **user-friendly** Python visualization library built on top of **Matplotlib** and **mplcursors**. It is designed to help you **explore large datasets smoothly** by implementing automatic data decimation and compression. Beyond performance, it provides a comprehensive suite of native mouse interactions (zoom, pan, annotate, measure) without requiring complex event-handling code.

## ✨ Core Features

- 🚀 **Smooth Large-Scale Plotting**: Automatically compresses and decimates data during rendering, ensuring fluid interaction even with datasets containing over 1,000,000 points.
- 🖱️ **Rich Native Interactions**: Full support for Left, Right, and Middle mouse button actions including zooming, panning, annotating, and measuring.
- 📏 **Built-in Tools**: Includes ruler tools, time-range sliders, color bars, and view history navigation.
- 📊 **Flexible Layouts**: Easy support for multi-subplot sharing (X/Y axes), logarithmic scales, and equal aspect ratios.
- 🏷️ **Smart Annotations**: Click to show/hide data info, click legends to toggle visibility, and custom data info tooltips.

## 🖱️ Interaction Guide

`yxsplot` comes with an intuitive default interaction scheme. No extra code is needed to enable these features:

| Action | Mouse / Keyboard | Description |
| :--- | :--- | :--- |
| **Zoom In** | Left Drag | Draw a box to zoom into a specific region. |
| **Reset View** | Right Click | Reset the plot to the original full view. |
| **Pan** | Right Drag | Move the view horizontally or vertically. |
| **Zoom Scroll** | Middle Wheel | Zoom in/out centered on the cursor. |
| **Auto-Scale Y** | Right Long Press | Automatically adjust the Y-axis scale to fit visible data. |
| **Show Data Info** | Left Click (on point) | Display data info for the specific point. |
| **Hide Data Info** | Right Click (on data info) | Remove the displayed data info. |
| **Toggle Curve** | Left Click (on legend) | Show or hide the corresponding curve. |
| **Ruler Tool** | Middle Drag | Draw a ruler to measure distance between points. |
| **Clear Ruler** | Middle Click | Clear all active rulers. |
| **History Back** | Keyboard ← | Go to the previous view state. |
| **History Forward**| Keyboard → | Go to the next view state. |
| **Reload Uncompressed View** | Left Click  ↻ | Reload and render the view without data compression (may be time-consuming). |

## 📦 Installation

```bash
pip install yxsplot
```

> **Requirements**: Python 3.12+, numpy, matplotlib, mplcursors.

## 🚀 Quick Start Examples

### 1. Basic Plotting with Large-Scale Data

Get started with a simple sine wave. Notice how `yxsplot` handles 1 million points effortlessly. You can also apply masks to filter data dynamically.

```python
import numpy as np
from yxsplot import plot, show_figure

# Generate 1 million points
n = 1_000_000
x = np.linspace(0, 2 * np.pi, n)
y = np.sin(x)

# Create an interactive plot
# 'mask' allows you to filter data visually without changing the source array
ax = plot(
    x,
    y,
    data_name="Sine Wave (y > -0.5)",
    x_name="Time (s)",
    y_name="Amplitude",
    title="Basic Plotting Demo (1M Points)",
    mask=y > -0.5,  # Only display points where y > -0.5
)

# Show the plot window
show_figure()
```

### 2. Multi-Plots with Shared Axes

Easily create complex dashboards with shared X/Y axes. `yxsplot` synchronizes interactions across plots automatically.

```python
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
```

### 3. Advanced Features: Log Scale, Time Range & Data Info

Explore advanced capabilities like logarithmic scales, time-range sliders, and custom data information tooltips.

```python
import numpy as np
from yxsplot import plot, show_figure

# Generate 1 million points for a unit circle
n = 1_000_000
t = np.linspace(0, 2 * np.pi, n)
x = np.cos(t)
y = np.sin(t)

# Prepare custom metadata to display in interactive data info.
# Keys become data info, values must be arrays of length n.
data_info = {
    "Category": ["Circle"] * n,      # Constant string for all points
    "x + y": x + y,                  # Custom value
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
```

## 🛠️ Why yxsplot?

Standard Matplotlib scripts often struggle with large datasets, becoming sluggish when interacting with more than 100k points. `yxsplot` improves this experience by:

1. **Intelligent Decimation**: Automatically reducing point density based on screen resolution while preserving peak values.
2. **Unified Interaction Model**: Combining zoom, pan, annotation, and measurement into a single, consistent mouse interface.
3. **Developer Friendly**: A simple functional API that hides the complexity of event connections and blitting.

## 📄 License

This project is licensed under the **MIT License**.

Copyright (c) 2026-present Xiaosheng Yang

## 🙋‍♂️ Author

- **Xiaosheng Yang**

🐛 [Open an Issue](https://github.com/yangxiaosheng27/yxsplot/issues) for bugs or features  
🔗 [GitHub Repository](https://github.com/yangxiaosheng27/yxsplot)

---

## 中文

`yxsplot` 是一个**用户友好型**的 Python 可视化库，基于 **Matplotlib** 和 **mplcursors** 构建。它旨在通过自动数据降采样和压缩技术，帮助你**流畅地探索大规模数据集**。除了性能优化，它还提供了全套原生的鼠标交互功能（缩放、平移、标注、测量），无需编写复杂的事件处理代码。

## ✨ 核心特性

- 🚀 **流畅的大规模绘图**：在渲染时自动压缩和降采样数据，即使面对超过 100 万个数据点，也能确保交互流畅。
- 🖱️ **丰富的原生交互**：全面支持鼠标左、右、中键操作，包括区域缩放、视图平移、数据标注和距离测量。
- 📏 **内置实用工具**：包含标尺工具、时间范围滑块、颜色条以及视图历史导航功能。
- 📊 **灵活的布局**：轻松支持多子图坐标轴共享（X/Y 轴）、对数刻度以及等比例显示。
- 🏷️ **智能标注**：点击数据点显示/隐藏数据信息，点击图例切换曲线可见性，并支持自定义数据信息提示框。

## 🖱️ 交互指南

`yxsplot` 自带一套直观的默认交互方案，无需额外代码即可直接使用：

| 操作 | 鼠标 / 键盘 | 描述 |
| :--- | :--- | :--- |
| **框选视图** | 左键拖拽 | 绘制矩形框以放大特定区域。 |
| **重置视图** | 右键单击 | 将视图重置为初始的全景状态。 |
| **平移视图** | 右键拖拽 | 水平或垂直移动视图。 |
| **缩放视图** | 中键滚轮 | 以光标为中心进行放大/缩小。 |
| **纵轴自适应** | 右键长按 | 自动调整 Y 轴刻度以适配当前可见数据。 |
| **显示数据信息** | 左键单击 (数据点) | 显示该数据点的详细数据信息。 |
| **隐藏数据信息** | 右键单击 (数据信息) | 移除已显示的数据信息。 |
| **开关曲线显示** | 左键单击 (图例) | 显示或隐藏对应的曲线。 |
| **标尺工具** | 中键拖拽 | 绘制标尺以测量两点间的距离。 |
| **清除标尺** | 中键单击 | 清除所有激活的标尺。 |
| **历史视图后退** | 键盘 ← | 返回上一个视图状态。 |
| **历史视图前进** | 键盘 → | 进入下一个视图状态。 |
| **重载无压缩视图** | 左键点击 ↻ | 重新加载并渲染无数据压缩的视图（可能很耗时）。 |

## 📦 安装

```bash
pip install yxsplot
```

> **环境要求**: Python 3.12+, numpy, matplotlib, mplcursors.

## 🚀 快速开始示例

### 1. 百万级数据的基础绘图

从一个简单的正弦波开始。注意 `yxsplot` 如何轻松处理 100 万个数据点。你还可以使用掩码（mask）动态过滤数据。

```python
import numpy as np
from yxsplot import plot, show_figure

# 生成 100 万个数据点
n = 1_000_000
x = np.linspace(0, 2 * np.pi, n)
y = np.sin(x)

# 创建交互式图表
# 'mask' 参数允许你在不修改原始数据数组的情况下，通过布尔索引进行视觉过滤。
# 只有满足条件的点会被渲染，这能显著提高大数据量下的局部查看性能或突出特定区域。
ax = plot(
    x,
    y,
    data_name="正弦波（y > -0.5）",
    x_name="时间 (s)",
    y_name="振幅",
    title="基础绘图演示 (100 万数据点)",
    mask=y > -0.5,  # 仅显示 y > -0.5 的数据点
)

# 显示绘图窗口
show_figure()
```

### 2. 共享轴的多图

轻松创建带有共享X/Y轴的复杂绘图。`yxsplot`会自动同步各图之间的交互。

```python
import numpy as np
from yxsplot import plot, show_figure

# 生成 100 万个数据点
n = 1_000_000
x = np.linspace(0, 2 * np.pi, n)
y1 = np.sin(x)
y2 = np.cos(x)

# --- 图 1: 基础正弦 (主视图) ---
# 创建第一个图形窗口和坐标轴。它将作为其他图表共享坐标轴的“主控端”。
ax1 = plot(
    x,
    y1,
    data_name="正弦",
    title="同轴叠加与联动演示 (主视图)",
)

# --- 图 2: 同轴叠加曲线 (非子图) ---
# 设置 'new_fig=False' 将此曲线绘制在与 ax1 **完全相同** 的坐标轴上。
# 新增另一条曲线，与上一条曲线共享同一套 X/Y 轴刻度和范围，在同一个画布上显示。
ax2 = plot(
    x, 
    y2, 
    data_name="余弦", 
    new_fig=False,  # 关键参数：不建新窗口/新坐标轴，直接叠加到当前 ax1 上
)

# --- 图 3: 完全联动视图 (同步缩放/平移) ---
# 这将创建一个 **新** 的独立图形窗口，但将其坐标轴与 ax1 绑定。
# 当你在 'ax1' (第一个窗口) 中进行缩放或平移时，'ax3' (这个窗口) 会完全同步更新。
ax3 = plot(
    x,
    y1,
    data_name="正弦色谱",
    share_x=ax1,      # 联动 X 轴：ax1 的缩放/平移会自动同步到 ax3 的 X 轴视图
    share_y=ax1,      # 联动 Y 轴：ax1 的缩放/平移会自动同步到 ax3 的 Y 轴视图
    title="联动视图",
    color=y1,         # 启用色谱：根据 Y 值显示颜色梯度
    color_name="数值",
    # 注意：由于未设置 new_fig=False，这将在一个独立的窗口中打开，
    # 但该窗口会与第一个窗口保持完美的视图同步。适合对比观察（如：原始波形 vs 热力图）。
)

# 渲染并显示所有已创建的交互式窗口
# 运行后将弹出两个窗口：
# 1. 【叠加图】：同一个坐标系内同时显示正弦和余弦两条曲线。
# 2. 【联动图】：独立的色谱图，其视图范围随第 1 个窗口实时同步。
show_figure()
```

### 3. 高级功能：对数刻度、时间范围与数据信息

探索对数刻度、时间范围滑块以及自定义数据信息提示框等高级功能。

```python
import numpy as np
from yxsplot import plot, show_figure

# 生成 100 万个点来构建一个单位圆
n = 1_000_000
t = np.linspace(0, 2 * np.pi, n)
x = np.cos(t)
y = np.sin(t)

# 准备自定义元数据，用于在交互式数据信息框中显示。
# 字典的键将作为数据信息，值必须是长度为 n 的数组。
data_info = {
    "类别": ["圆形"] * n,          # 所有点共有的常量字符串
    "x + y": x + y,               # 自定义数值
    "标记 (y > 0)": y > 0,         # 布尔掩码（例如：区分上半圆，True/False）
}

# --- 图 1: 对数坐标与自定义标记 ---
# 使用 exp() 变换坐标以演示对数刻度的可视化效果。
# 由于 x 和 y 来自单位圆 (-1 到 1)，exp(x) 和 exp(y) 将保证为正数，适合对数坐标。
# 注意：此处未设置 new_fig=False，因此这将弹出一个【独立】的新窗口。
ax1 = plot(
    np.exp(x),                # X 数据（变换后）
    np.exp(y),                # Y 数据（变换后）
    data_name="指数圆",        # 图例标签
    x_name="log x",           # X 轴标签
    y_name="log y",           # Y 轴标签
    log_x=True,               # 启用 X 轴对数刻度
    log_y=True,               # 启用 Y 轴对数刻度
    line_marker="*",          # 使用星号标志，而非默认的线点
    title="高级功能：对数刻度、自定义线条标志",
)

# --- 图 2: 等比例约束、色谱与数据信息 ---
# 绘制原始圆形，并添加更多交互功能。
# 注意：同样未设置 new_fig=False，因此这将弹出【第二个独立】的新窗口。
ax2 = plot(
    x,                        # 原始 X 数据
    y,                        # 原始 Y 数据
    x_name="x",               # X 轴标签
    y_name="y",               # Y 轴标签
    color=t,                  # 将参数 't' (角度) 映射为颜色
    color_name="t",           # 颜色条的标签
    data_info=data_info,      # 绑定上方定义的自定义提示框数据
    equal_scale=True,         # 强制 1:1 纵横比（对于绘制正圆至关重要，否则圆会变椭圆）
    time_range=[1, 5],        # 初始化时间范围滑块的默认显示窗口
    title="高级功能：等比例约束、自定义数据信息",
)

# 渲染并显示所有已创建的交互式图窗
# 由于上面创建了两个独立的图，这里会同时弹出两个窗口
show_figure()
```

## 🛠️ 为什么选择 yxsplot？

标准的 Matplotlib 脚本在处理大数据集时往往表现吃力，当数据点超过 10 万时交互容易变得卡顿。`yxsplot` 通过以下方式改善这一体验：

1. **智能降采样**：根据屏幕分辨率自动降低点密度，同时保留峰值特征。
2. **统一的交互模型**：将缩放、平移、标注和测量整合到一个一致且直观的鼠标操作界面中。
3. **开发者友好**：提供简洁的函数式 API，隐藏了事件连接和 blitting 的复杂性。

## 📄 许可证

本项目采用 **MIT License** 授权。

Copyright (c) 2026-present Xiaosheng Yang

## 🙋‍♂️ 作者

- **杨晓生**

🐛 [提交 Issue](https://github.com/yangxiaosheng27/yxsplot/issues) 反馈问题或建议  
🔗 [GitHub 仓库](https://github.com/yangxiaosheng27/yxsplot)
