"""
Module: yxsplot
Summary: Smooth and interactive 2D plotting for large-scale datasets.
Author: Xiaosheng Yang
Date: 2026/2/18
Environment: python(3.12.2~3.13.12), numpy(1.26.4~2.4), matplotlib(3.8.0~3.10), mplcursors(0.5.3~0.7.1)
"""

__all__ = ["plot", "show_figure", "close_figure", "welch"]

_DEBUG = False
_LANGUAGE = "auto"

# import matplotlib
# matplotlib.use("QtAgg")

from matplotlib.collections import LineCollection
from matplotlib.collections import PathCollection
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import RangeSlider
from matplotlib.patches import PathPatch
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.colors import Colormap
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.path import Path
from matplotlib.axes import Axes
from types import NoneType
import numpy as np
import mplcursors
import warnings
import numbers
import locale
import time


def _enable_debug():
    global _DEBUG
    _DEBUG = True


def _disable_debug():
    global _DEBUG
    _DEBUG = False


def _set_language(language):
    global _LANGUAGE
    _LANGUAGE = str(language).lower()


def _get_language():
    global _LANGUAGE
    return str(_LANGUAGE).lower()


def _check_language_Chinese():
    try:
        if _get_language() == "auto":
            lang, _ = locale.getlocale()
            if lang:
                if lang.startswith("Chinese") or lang.startswith("zh"):
                    return True
        elif _get_language() == "chinese":
            return True
    except:
        pass
    return False


def _debug_print(*args, **kwargs):
    global _DEBUG
    if _DEBUG:
        print(*args, **kwargs)


def _get_limit(x_min, x_max, log_x, margin=0, data_range=None):
    if not log_x:
        if not data_range:
            data_range = x_max - x_min
        else:
            data_mid = (x_min + x_max) / 2
            x_min = data_mid - data_range / 2
            x_max = data_mid + data_range / 2
        delta = data_range * margin
        if delta < 1e-10:
            delta = x_max * margin
        if delta < 1e-10:
            delta = margin
        x_limit = [x_min - delta, x_max + delta]
    else:
        if not data_range:
            x_min = max(x_min, 1e-10)
            x_max = max(x_max, 1e-10)
            data_range = np.log10(x_max) - np.log10(x_min)
        else:
            data_mid = (np.log10(x_min) + np.log10(x_max)) / 2
            x_min = 10 ** (data_mid - data_range / 2)
            x_max = 10 ** (data_mid + data_range / 2)
            x_min = max(x_min, 1e-10)
            x_max = max(x_max, 1e-10)
        delta = data_range * margin
        if delta < 1e-10:
            delta = np.log10(x_max) * margin
        if delta < 1e-10:
            delta = margin
        new_x_min = 10 ** (np.log10(x_min) - delta)
        new_x_max = 10 ** (np.log10(x_max) + delta)
        x_limit = [new_x_min, new_x_max]
    return x_limit


def _get_equal_scale_limit(x_min, x_max, y_min, y_max, log_x, log_y, margin=0):
    x_range = (x_max - x_min) if not log_x else (np.log10(x_max) - np.log10(x_min))
    y_range = (y_max - y_min) if not log_y else (np.log10(y_max) - np.log10(y_min))
    data_range = max(x_range, y_range)
    x_limit = _get_limit(x_min, x_max, log_x, data_range=data_range, margin=margin)
    y_limit = _get_limit(y_min, y_max, log_y, data_range=data_range, margin=margin)
    return x_limit, y_limit


def _auto_scale(ax, scale_x=True, scale_y=True, margin=0.05):
    t0 = time.time()
    equal_scale = ax.my_data["equal_scale"]
    scale_update_flag = False
    if not scale_x and not scale_y:
        return scale_update_flag

    def get_raw_x_y(artist):
        x = artist.my_data["raw_x"]
        y = artist.my_data["raw_y"]
        ax = artist.axes
        fig = ax.figure
        if hasattr(fig, "my_timeslider"):
            length = len(x)
            valmin = fig.my_timeslider.valmin
            valmax = fig.my_timeslider.valmax
            valrange = valmax - valmin
            timeslider_min, timeslider_max = fig.my_timeslider.val
            timeslider_min = (timeslider_min - valmin) / valrange
            timeslider_max = (timeslider_max - valmin) / valrange
            time_valid_mask = np.zeros(length, dtype=bool)
            time_count_min = max(int(timeslider_min * length), 0)
            time_count_max = min(int(timeslider_max * length), length)
            time_valid_mask[time_count_min:time_count_max] = True
            x = x[time_valid_mask]
            y = y[time_valid_mask]
        return x, y

    x_vals = []
    y_vals = []
    for artist in ax.get_children():
        if not artist.get_visible() or not isinstance(artist, (Line2D, PathCollection)):
            continue
        if isinstance(artist, Line2D):  #  from plot()
            if hasattr(artist, "my_data"):
                x, y = get_raw_x_y(artist)
            else:
                x, y = artist.get_data()
        elif isinstance(artist, PathCollection):  #  from scatter()
            offsets = artist.get_offsets()  # shape (N, 2)
            if offsets.size == 0:
                continue
            if hasattr(artist, "my_data"):
                x, y = get_raw_x_y(artist)
            else:
                x, y = offsets[:, 0], offsets[:, 1]
        if scale_x and scale_y:
            x_vals.append(x)
            y_vals.append(y)
        elif scale_x and not scale_y:
            ylim = ax.get_ylim()
            _mask = (y >= ylim[0]) & (y <= ylim[1])
            if np.any(_mask):
                x_vals.append(x[_mask])
                if equal_scale:
                    y_vals.append(y[_mask])
        elif not scale_x and scale_y:
            xlim = ax.get_xlim()
            _mask = (x >= xlim[0]) & (x <= xlim[1])
            if np.any(_mask):
                y_vals.append(y[_mask])
                if equal_scale:
                    x_vals.append(x[_mask])

    # get x_min, x_max
    if x_vals:
        x_all = np.concatenate(x_vals)
        x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
        if not np.all(np.isfinite([np.nanmin(x_all), np.nanmax(x_all)])):
            x_min, x_max = None, None
    else:
        x_min, x_max = None, None

    # get y_min, y_max
    if y_vals:
        y_all = np.concatenate(y_vals)
        y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
        if not np.all(np.isfinite([np.nanmin(y_all), np.nanmax(y_all)])):
            y_min, y_max = None, None
    else:
        y_min, y_max = None, None

    # get log_x, log_y
    log_x = ax.xaxis.get_scale() in ["log", "symlog"]
    log_y = ax.yaxis.get_scale() in ["log", "symlog"]

    # set x_limit, y_limit
    x_limit_enable = x_min is not None and x_max is not None
    y_limit_enable = y_min is not None and y_max is not None
    if equal_scale and x_limit_enable and y_limit_enable:
        x_limit, y_limit = _get_equal_scale_limit(
            x_min, x_max, y_min, y_max, log_x, log_y, margin=margin
        )
        ax.set(xlim=(x_limit[0], x_limit[1]), ylim=(y_limit[0], y_limit[1]))
    else:
        if x_limit_enable:
            x_limit = _get_limit(x_min, x_max, log_x, margin=margin)
            ax.set_xlim(x_limit[0], x_limit[1])
        if y_limit_enable:
            y_limit = _get_limit(y_min, y_max, log_y, margin=margin)
            ax.set_ylim(y_limit[0], y_limit[1])

    if getattr(ax, "my_data", None) and ax.my_data["ax_range"] != (
        ax.get_xlim(),
        ax.get_ylim(),
    ):
        scale_update_flag = True
        t1 = time.time()
        if t1 - t0 > 0.01:
            _debug_print(
                f"fig({ax.figure.number}) _auto_scale(): {(t1 - t0) * 1e3:g} ms"
            )
    return scale_update_flag


def _push_ax(ax):
    def get_share_figure(ax):
        try:
            shared_x_figure = [
                share_axes.figure
                for share_axes in ax.get_shared_x_axes().get_siblings(ax)
            ]
            shared_y_figure = [
                share_axes.figure
                for share_axes in ax.get_shared_y_axes().get_siblings(ax)
            ]
            return set(shared_x_figure) | set(shared_y_figure)
        except:
            return [ax.figure]

    for share_fig in get_share_figure(ax):
        share_fig.canvas.manager.toolbar.push_current()  # push current fig into stack


def _call_back_on_add_cursor(sel):
    if not sel.index:
        return
    # get index
    index = sel.index[0] if isinstance(sel.index, tuple) else sel.index
    index = round(index)
    # set bbox color
    bbox = sel.annotation.get_bbox_patch()
    if hasattr(sel.artist, "get_color"):
        color = sel.artist.get_color()
        if hasattr(color, "ndim") and color.ndim == 2:
            color = color[index]
    elif hasattr(sel.artist, "get_facecolor"):
        color = sel.artist.get_facecolor()[index]
    else:
        color = "#A4A4A48F"
    bbox.set_facecolor(color)
    bbox.set_alpha(0.5)
    # set arrowprops
    arrowprops = sel.annotation.arrow_patch
    arrowprops.set_arrowstyle("-")
    arrowprops.set_alpha(0.5)
    # set text
    if hasattr(sel.artist, "my_data"):
        # get label
        label = sel.artist.get_label()
        if label and not label.startswith("_"):
            text = f"[{label}]\n"
        else:
            text = ""
        # get raw_x, raw_y, raw_color, raw_index
        raw_x = sel.artist.my_data["raw_x"]
        raw_y = sel.artist.my_data["raw_y"]
        raw_color = sel.artist.my_data["raw_color"]
        raw_index = np.flatnonzero(sel.artist.my_data["compress_data_mask"])[index]
        text += f"({raw_x[raw_index]:g}, {raw_y[raw_index]:g})"
        if len(raw_color):
            text += f"\ncolor: {raw_color[raw_index]:g}"
        # get data_info
        data_info = sel.artist.my_data["data_info"]
        if data_info is not None:
            for key in data_info:
                value = data_info[key][raw_index]
                if isinstance(value, numbers.Number) and not isinstance(value, bool):
                    text += f"\n{key}: {value:g}"
                else:
                    text += f"\n{key}: {value}"
        sel.annotation.set_text(text)
    # set horizont alalignment
    sel.annotation.set_ha("left")


def _disable_all_cursor_drag(cursor):
    for cursor_ in cursor.selections:
        cursor_.annotation.draggable(False)


def _enable_all_cursor_drag(cursor):
    for cursor_ in cursor.selections:
        cursor_.annotation.draggable(True)


def _enable_single_cursor_drag(cursor, draggable_cursor):
    for cursor_ in cursor.selections:
        if cursor_ != draggable_cursor:
            cursor_.annotation.draggable(False)
        else:
            return True
    return False


def _call_back_on_pick(event):
    if event.mouseevent.inaxes:
        ax = event.mouseevent.inaxes
    elif event.canvas.figure.axes and len(event.canvas.figure.axes):
        ax = event.canvas.figure.axes[0]
    else:
        ax = None
    if not hasattr(ax, "my_data"):
        return
    ax.my_data["cursor"]._on_pick(event)
    if event.artist is ax.my_data["help_button"]:
        ax.my_data["help_text"].set_visible(not ax.my_data["help_text"].get_visible())
        ax.figure.canvas.draw_idle()
    elif event.artist is ax.my_data["full_load_button"]:
        ax.my_data["full_load_button"].set_visible(False)
        ax.my_data["full_load"] = True
        ax.figure.canvas.draw_idle()
    ax.figure.my_data["current_ax"] = ax


def _pick_cursor(event, ax):
    for cursor in reversed(ax.my_data["cursor"].selections):
        ann = cursor.annotation
        if isinstance(ann, Text) and ann.get_visible():
            ann_bbox = ann.get_window_extent(renderer=ax.figure.canvas.get_renderer())
            ax_bbox = ax.get_window_extent()
            if ax_bbox.overlaps(
                ann_bbox
            ):  # Check if ann is actually rendered and visible on screen.
                patch_bbox = ann.get_bbox_patch().get_window_extent()
                tol = 0
                if ((patch_bbox.xmin - tol) <= event.x <= (patch_bbox.xmax + tol)) and (
                    (patch_bbox.ymin - tol) <= event.y <= (patch_bbox.ymax + tol)
                ):
                    return cursor
    return None


def _pick_legend(event):
    ax = event.inaxes
    if ax:
        legend = ax.get_legend()
        if legend and legend.get_window_extent().contains(event.x, event.y):
            return legend
    return None


def _legend_switch_visible(legend, pick_x, pick_y):
    if not legend:
        return
    bbox = legend.get_window_extent()
    y = (pick_y - bbox.ymin) / bbox.height
    n = len(legend.legend_handles)
    artists = []
    for artist in legend.axes.get_lines():
        if artist.get_label() and artist.get_label()[0] != "_":
            artists.append(artist)
    for artist in legend.axes.collections:
        if artist.get_label() and artist.get_label()[0] != "_":
            artists.append(artist)
    assert n == len(artists), f"len(legend)={n}, but len(artists)={len(artists)}!"
    if n == 0:
        return
    index = int((1.0 - y) // (1.0 / n))
    if not 0 <= index < n:
        return
    artist = artists[index]
    if artist.get_visible() is True:
        artist.set_visible(False)
        _debug_print(
            f"fig({artist.axes.figure.number}), ax({id(artist.axes) % 10000:04d}), artist({id(artist) % 10000:04d}) set_visible(False)",
        )
        if isinstance(artist, PathCollection) and hasattr(artist, "my_line"):
            artist.my_line.set_visible(False)
        legend.legend_handles[index].set_alpha(0.2)
    else:
        artist.set_visible(True)
        _debug_print(
            f"fig({artist.axes.figure.number}), ax({id(artist.axes) % 10000:04d}), artist({id(artist) % 10000:04d}) set_visible(True)",
        )
        if isinstance(artist, PathCollection) and hasattr(artist, "my_line"):
            artist.my_line.set_visible(True)
        legend.legend_handles[index].set_alpha(1.0)
    legend.figure.canvas.draw_idle()


def _call_back_on_button_press(event):

    t0 = time.time() / 1e3
    ax = event.inaxes or getattr(event.canvas.figure, "my_data", {}).get("current_ax")
    if not hasattr(ax, "my_data"):
        return
    ax.figure.my_data["current_ax"] = ax
    # init
    ax.my_data["rectangle_select_is_busy"] = False
    ax.my_data["cursor_select_is_busy"] = False
    ax.my_data["cursor_drag_is_busy"] = False
    ax.my_data["measure_is_busy"] = False
    ax.my_data["picked_cursor"] = None
    ax.my_data["picked_legend"] = None
    ax.my_data["rectangle_select_start_point"] = None
    ax.my_data["measure_start_point"] = None
    cursor = ax.my_data["cursor"]
    # cursor drag
    if event.button != MouseButton.LEFT:
        _disable_all_cursor_drag(cursor)
    if event.button == MouseButton.LEFT:
        # time
        ax.my_data["left_button_pressed_time"] = time.time()
        # pick
        ax.my_data["picked_cursor"] = _pick_cursor(event, ax)
        if ax.my_data["picked_cursor"]:
            ax.my_data["cursor_drag_is_busy"] = _enable_single_cursor_drag(
                cursor, ax.my_data["picked_cursor"]
            )
        else:
            _disable_all_cursor_drag(cursor)
            ax.my_data["picked_legend"] = _pick_legend(event)
        # legend handle
        if ax.my_data["picked_legend"]:
            _legend_switch_visible(ax.my_data["picked_legend"], event.x, event.y)
        # rectangle select
        if (
            not ax.my_data["picked_cursor"]
            and not ax.my_data["picked_legend"]
            and event.inaxes
        ):
            ax.my_data["rectangle_select_start_point"] = (event.x, event.y)
            ax.my_data["ax_background"] = ax.figure.canvas.copy_from_bbox(ax.bbox)
    elif event.button == MouseButton.RIGHT:
        # time
        ax.my_data["right_button_pressed_time"] = time.time()
        # pick
        ax.my_data["picked_cursor"] = _pick_cursor(event, ax)
        if not ax.my_data["picked_cursor"]:
            if ax.my_data["right_button_timer"]:
                ax.my_data["right_button_timer"].start()
        # pan
        event.button = MouseButton.LEFT  # for pan of toolbar
        ax.my_data["ax_range"] = ax.get_xlim(), ax.get_ylim()
        plt.get_current_fig_manager().toolbar.press_pan(event)
        event.button = MouseButton.RIGHT
    elif event.button == MouseButton.MIDDLE:
        if event.inaxes:
            for key in [
                "measure_patch",
                "measure_text_d",
                "measure_text_dx",
                "measure_text_dy",
            ]:
                ax.my_data[key].append(None)
            ax.my_data["measure_start_point"] = (event.x, event.y)
            ax.my_data["ax_background"] = ax.figure.canvas.copy_from_bbox(ax.bbox)
    t1 = time.time() / 1e3
    if t1 - t0 > 0.01:
        _debug_print(
            f"_call_back_on_button_press(): {t1 - t0}ms",
        )


def _measure_on_motion(event, ax):
    if not ax.my_data["measure_start_point"]:
        return
    ax.my_data["measure_is_busy"] = True
    x0, y0 = ax.my_data["measure_start_point"]
    x1, y1 = event.x, event.y
    tolerance = 10
    if abs(x0 - x1) < tolerance or abs(y0 - y1) < tolerance:
        return
    boundary = ax.get_window_extent()
    x1, y1 = (
        np.clip(x1, boundary.x0, boundary.x1),
        np.clip(y1, boundary.y0, boundary.y1),
    )
    data_x0, data_y0 = ax.transData.inverted().transform((x0, y0))
    data_x1, data_y1 = ax.transData.inverted().transform((x1, y1))
    verts = np.array(
        [
            [data_x0, data_y0],
            [data_x1, data_y1],  # d
            [data_x0, data_y0],
            [data_x1, data_y0],  # dx
            [data_x1, data_y0],
            [data_x1, data_y1],  # dy
        ]
    )
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.MOVETO,
        Path.LINETO,
        Path.MOVETO,
        Path.LINETO,
    ]
    if not ax.my_data["measure_patch"][-1]:
        ax.my_data["measure_patch"][-1] = ax.add_patch(
            PathPatch(
                Path([(0, 0)], [Path.MOVETO]),
                linestyle="--",
                lw=1,
                alpha=0.5,
                color="black",
                fill=False,
            )
        )
        ax.my_data["measure_text_d"][-1] = ax.text(0, 0, "", ha="center", va="center")
        ax.my_data["measure_text_dx"][-1] = ax.text(0, 0, "", ha="center", va="center")
        ax.my_data["measure_text_dy"][-1] = ax.text(0, 0, "", ha="center", va="center")
    ax.my_data["measure_patch"][-1].set_path(Path(verts, codes))
    dx, dy = data_x1 - data_x0, data_y1 - data_y0
    d = np.hypot(dx, dy)
    ax.my_data["measure_text_d"][-1].set_text(f"{d:g}")
    ax.my_data["measure_text_d"][-1].set_position(
        ((data_x0 + data_x1) / 2, (data_y0 + data_y1) / 2)
    )
    ax.my_data["measure_text_dx"][-1].set_text(f"{abs(dx):g}")
    ax.my_data["measure_text_dx"][-1].set_position(((data_x0 + data_x1) / 2, data_y0))
    ax.my_data["measure_text_dx"][-1].set_va("top" if dy > 0 else "bottom")
    ax.my_data["measure_text_dy"][-1].set_text(f"{abs(dy):g}")
    ax.my_data["measure_text_dy"][-1].set_position((data_x1, (data_y0 + data_y1) / 2))
    ax.my_data["measure_text_dy"][-1].set_ha("left" if dx > 0 else "right")
    ax.figure.canvas.restore_region(ax.my_data["ax_background"])
    ax.draw_artist(ax.my_data["measure_patch"][-1])
    ax.draw_artist(ax.my_data["measure_text_d"][-1])
    ax.draw_artist(ax.my_data["measure_text_dx"][-1])
    ax.draw_artist(ax.my_data["measure_text_dy"][-1])
    ax.figure.canvas.blit(ax.bbox)


def _rectangle_select_on_motion(event, ax):
    if not ax.my_data["rectangle_select_start_point"]:
        return
    x0, y0 = ax.my_data["rectangle_select_start_point"]
    x1, y1 = event.x, event.y
    tolerance = 5
    if abs(x0 - x1) < tolerance or abs(y0 - y1) < tolerance:
        if ax.my_data["rectangle_select"]:
            ax.my_data["rectangle_select"].set_visible(False)
            ax.figure.canvas.restore_region(ax.my_data["ax_background"])
            ax.draw_artist(ax.my_data["rectangle_select"])
            ax.figure.canvas.blit(ax.bbox)
        return
    ax.my_data["rectangle_select_is_busy"] = True
    boundary = ax.get_window_extent()
    x1, y1 = (
        np.clip(x1, boundary.x0, boundary.x1),
        np.clip(y1, boundary.y0, boundary.y1),
    )
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])
    data_x0, data_y0 = ax.transData.inverted().transform((x0, y0))
    data_x1, data_y1 = ax.transData.inverted().transform((x1, y1))
    if not ax.my_data["rectangle_select"]:
        ax.my_data["rectangle_select"] = ax.add_patch(
            Rectangle(
                (0, 0),
                0,
                0,
                linestyle="-",
                lw=1.5,
                alpha=0.3,
                color="black",
                fill=True,
            )
        )
    ax.my_data["rectangle_select"].set_xy((data_x0, data_y0))
    ax.my_data["rectangle_select"].set_width(data_x1 - data_x0)
    ax.my_data["rectangle_select"].set_height(data_y1 - data_y0)
    ax.my_data["rectangle_select"].set_visible(True)
    ax.figure.canvas.restore_region(ax.my_data["ax_background"])
    ax.draw_artist(ax.my_data["rectangle_select"])
    ax.figure.canvas.blit(ax.bbox)


def _call_back_on_motion(event):

    ax = event.inaxes or getattr(event.canvas.figure, "my_data", {}).get("current_ax")
    if not hasattr(ax, "my_data"):
        return
    if ax.my_data["cursor_select_is_busy"] or ax.my_data["cursor_drag_is_busy"]:
        return
    _measure_on_motion(event, ax)
    _rectangle_select_on_motion(event, ax)


def _call_back_on_button_release(event):
    t0 = time.time() / 1e3
    ax = event.inaxes or getattr(event.canvas.figure, "my_data", {}).get("current_ax")
    if not hasattr(ax, "my_data"):
        return
    cursor = ax.my_data["cursor"]
    ax.my_data["button_release_time"] = time.time()
    if event.button == MouseButton.LEFT:
        if (
            ax.my_data["left_button_pressed_time"] is not None
            and (
                ax.my_data["button_release_time"]
                - ax.my_data["left_button_pressed_time"]
            )
            < ax.my_data["mouse_button_hold_time_threshold"]
        ):
            # cursor select
            if (
                not ax.my_data["picked_legend"]
                and not ax.my_data["cursor_drag_is_busy"]
                and not ax.my_data["measure_is_busy"]
                and not ax.my_data["rectangle_select_is_busy"]
            ):
                ax.my_data["cursor_select_is_busy"] = True
                cursor._on_select_event(event)
        # rectangle select
        ax.my_data["rectangle_select_start_point"] = None
        if ax.my_data["rectangle_select"]:
            if ax.my_data["rectangle_select"].get_visible() == True:
                x0, y0 = ax.my_data["rectangle_select"].get_xy()
                w = ax.my_data["rectangle_select"].get_width()
                h = ax.my_data["rectangle_select"].get_height()
                x_min, x_max = x0, x0 + w
                y_min, y_max = y0, y0 + h
                if x_min < x_max and y_min < y_max:
                    if ax.my_data["equal_scale"]:
                        # get log_x, log_y
                        log_x = ax.xaxis.get_scale() in ["log", "symlog"]
                        log_y = ax.yaxis.get_scale() in ["log", "symlog"]
                        # set x_limit, y_limit
                        x_limit, y_limit = _get_equal_scale_limit(
                            x_min, x_max, y_min, y_max, log_x, log_y
                        )
                        ax.set(
                            xlim=(x_limit[0], x_limit[1]),
                            ylim=(y_limit[0], y_limit[1]),
                        )
                    else:
                        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
                    _push_ax(ax)
                ax.my_data["rectangle_select"].set_visible(False)
            ax.my_data["rectangle_select"] = None
            ax.figure.canvas.draw_idle()
    elif event.button == MouseButton.RIGHT:
        plt.get_current_fig_manager().toolbar.release_pan(event)
        if ax.my_data["right_button_timer"]:
            ax.my_data["right_button_timer"].stop()
        if (
            ax.my_data["ax_range"] == (ax.get_xlim(), ax.get_ylim())
            and ax.my_data["right_button_pressed_time"] is not None
        ):
            if ax.my_data["picked_cursor"]:
                cursor.remove_selection(ax.my_data["picked_cursor"])
            else:
                if (
                    ax.my_data["button_release_time"]
                    - ax.my_data["right_button_pressed_time"]
                ) < ax.my_data["mouse_button_hold_time_threshold"]:
                    scale_update_flag = _auto_scale(ax, scale_x=True, scale_y=True)
                    if scale_update_flag:
                        _push_ax(ax)
                        ax.figure.canvas.draw_idle()
    elif event.button == MouseButton.MIDDLE:
        ax.my_data["measure_start_point"] = None
        measure_remove_flag = False
        if len(ax.my_data["measure_patch"]) > 1 and not ax.my_data["measure_patch"][-1]:
            for key in [
                "measure_patch",
                "measure_text_d",
                "measure_text_dx",
                "measure_text_dy",
            ]:
                for artist in ax.my_data[key]:
                    if artist:
                        artist.remove()
                        measure_remove_flag = True
                ax.my_data[key] = []
        if measure_remove_flag:
            ax.figure.canvas.draw_idle()
    # cursor drag
    _enable_all_cursor_drag(cursor)
    # reset
    ax.figure.my_data["current_ax"] = None
    ax.my_data["left_button_pressed_time"] = None
    ax.my_data["right_button_pressed_time"] = None
    ax.my_data["picked_cursor"] = None
    ax.my_data["picked_legend"] = None
    ax.my_data["rectangle_select_is_busy"] = False
    ax.my_data["cursor_select_is_busy"] = False
    ax.my_data["cursor_drag_is_busy"] = False
    ax.my_data["measure_is_busy"] = False
    t1 = time.time() / 1e3
    if t1 - t0 > 0.01:
        _debug_print(
            f"_call_back_on_button_release(): {t1 - t0}ms",
        )


def _call_back_on_right_button_timeout():
    fig = plt.gcf()
    if not fig:
        return
    axes = fig.get_axes()
    if len(axes) <= 0:
        return
    ax = axes[0]
    if ax.my_data["right_button_pressed_time"] is None:
        return
    if ax.my_data["ax_range"] != (ax.get_xlim(), ax.get_ylim()):  # in pan mode
        return
    plt.get_current_fig_manager().toolbar.release_pan(None)
    scale_update_flag = _auto_scale(ax, scale_x=False, scale_y=True)
    if scale_update_flag:
        _push_ax(ax)
        ax.figure.canvas.draw_idle()


def _call_back_on_scroll(event):
    if event.button != "up" and event.button != "down":
        return
    ax = event.inaxes or getattr(event.canvas.figure, "my_data", {}).get("current_ax")
    if not hasattr(ax, "my_data"):
        return
    if ax.my_data["measure_is_busy"]:
        return
    if event.inaxes:
        if ax.xaxis.get_scale() not in ["log", "symlog"]:
            x_min, x_max = ax.get_xlim()
        else:
            x_min, x_max = np.log10(ax.get_xlim())
        if ax.yaxis.get_scale() not in ["log", "symlog"]:
            y_min, y_max = ax.get_ylim()
        else:
            y_min, y_max = np.log10(ax.get_ylim())
        x_center = (x_min + x_max) / 2.0  # or x_center = event.xdata
        y_center = (y_min + y_max) / 2.0  # or y_center = event.ydata
        k = 0.1
        factor = (1.0 - k) if event.button == "up" else (1.0 + k)
        x_min = (x_min - x_center) * factor + x_center
        x_max = (x_max - x_center) * factor + x_center
        y_min = (y_min - y_center) * factor + y_center
        y_max = (y_max - y_center) * factor + y_center
        if ax.xaxis.get_scale() not in ["log", "symlog"]:
            ax.set(xlim=(x_min, x_max))
        else:
            ax.set(xlim=(10**x_min, 10**x_max))
        if ax.yaxis.get_scale() not in ["log", "symlog"]:
            ax.set(ylim=(y_min, y_max))
        else:
            ax.set(ylim=(10**y_min, 10**y_max))
        _push_ax(ax)
    cursor = ax.my_data["cursor"]
    # disable cursor drag
    _disable_all_cursor_drag(cursor)
    if event.inaxes:
        ax.figure.canvas.draw_idle()
    # enable cursor drag
    _enable_all_cursor_drag(cursor)


def _update_compress_data(
    ax,
    compress_length=2000,
    zoom_in_factor=2,
    zoom_out_factor=2,
    full_load=False,
):

    def check_need_update_compress(compress_param, last_compress_param):
        if not last_compress_param or not compress_param:
            return True
        offset = 6
        if not np.allclose(last_compress_param[offset:], compress_param[offset:]):
            return True
        (
            last_ax_range_x_min,
            last_ax_range_x_max,
            last_ax_range_y_min,
            last_ax_range_y_max,
            last_ax_range_x,
            last_ax_range_y,
        ) = last_compress_param[:offset]
        (
            ax_range_x_min,
            ax_range_x_max,
            ax_range_y_min,
            ax_range_y_max,
            ax_range_x,
            ax_range_y,
        ) = compress_param[:offset]
        if (
            ax_range_x < last_ax_range_x / zoom_in_factor
            or ax_range_x > last_ax_range_x * zoom_out_factor
        ):
            return True
        if (
            ax_range_y < last_ax_range_y / zoom_in_factor
            or ax_range_y > last_ax_range_y * zoom_out_factor
        ):
            return True
        if ax_range_x_min < last_ax_range_x_min - last_ax_range_x * zoom_out_factor / 2:
            return True
        if ax_range_x_max > last_ax_range_x_max + last_ax_range_x * zoom_out_factor / 2:
            return True
        if ax_range_y_min < last_ax_range_y_min - last_ax_range_y * zoom_out_factor / 2:
            return True
        if ax_range_y_max > last_ax_range_y_max + last_ax_range_y * zoom_out_factor / 2:
            return True
        return False

    def get_compressed_data_mask(
        x,
        y,
        ax_range_x,
        ax_range_y,
        ax_width_pixel,
        ax_height_pixel,
        max_compress_pixel=20,
        min_compress_pixel=1,
        valid_data_mask=None,
    ):

        def point_to_segment_distance(points):
            P, A, B = points[2:], points[:-2], points[1:-1]
            AB, AP = B - A, P - A
            dot = np.einsum("ij,ij->i", AP, AB, optimize=True)
            len_sq = np.einsum("ij,ij->i", AB, AB, optimize=True)
            t = np.clip(dot / np.where(len_sq == 0, 1.0, len_sq), 0.0, 1.0)
            C = A + t[:, None] * AB
            return np.sqrt(np.einsum("ij,ij->i", P - C, P - C, optimize=True))

        def compress_handle(compress_pixel, method=1):
            if method == 0:
                d = np.linalg.norm(np.diff(points, axis=0), axis=1)
            else:
                d = point_to_segment_distance(points)
            d = np.where(np.isfinite(d), d, 0)
            sum_d = np.cumsum(d)
            valid_pixel_mask = np.diff(sum_d // compress_pixel) > 0
            compress_data_mask = np.concatenate(
                [
                    valid_data_mask[: (length - len(valid_pixel_mask))],
                    valid_pixel_mask,
                ]
            )
            compress_valid_length = np.count_nonzero(compress_data_mask)
            return compress_data_mask, compress_valid_length

        assert max_compress_pixel >= min_compress_pixel
        length = len(x)
        try:
            if valid_data_mask is None:
                valid_data_mask = np.isfinite(x) & np.isfinite(y)

            x *= ax_width_pixel / ax_range_x
            y *= ax_height_pixel / ax_range_y
            points = np.stack((x, y), axis=1)

            if length > 2_000_000:
                compress_pixel = max_compress_pixel
                compress_data_mask, compress_valid_length = compress_handle(
                    compress_pixel
                )
            else:
                compress_pixel = min_compress_pixel
                compress_data_mask, compress_valid_length = compress_handle(
                    min_compress_pixel
                )
                if compress_valid_length > compress_length:
                    compress_pixel = int(
                        min_compress_pixel * compress_valid_length / compress_length
                    )
                    compress_pixel = min(compress_pixel, max_compress_pixel)
                    if compress_pixel > min_compress_pixel:
                        compress_data_mask, compress_valid_length = compress_handle(
                            compress_pixel
                        )
            compress_data_mask[:-1] = compress_data_mask[:-1] | (
                valid_data_mask[:-1] & ~valid_data_mask[1:]
            )
            compress_data_mask[-1] = valid_data_mask[-1]
        except Exception as e:
            print("\ncompress_data: %s" % str(e))
            compress_data_mask = np.ones(length, dtype=bool)
            compress_valid_length = length
            compress_pixel = 0
        return compress_data_mask, compress_valid_length, compress_pixel

    def trim_out_range_mask(mask, trim=1):
        from numpy.lib.stride_tricks import sliding_window_view

        mask = np.asarray(mask, dtype=bool)
        n = len(mask)
        window = 2 * trim + 1
        if n < window:
            return np.zeros_like(mask, dtype=bool)
        windows = sliding_window_view(mask, window)
        valid = np.all(windows, axis=1)
        trim_mask = np.zeros_like(mask, dtype=bool)
        trim_mask[trim:-trim] = valid
        return trim_mask

    ax_full_load_state = True
    if ax and hasattr(ax, "my_data"):
        for artist in ax.get_children():
            if not artist.get_visible() or not isinstance(
                artist, (Line2D, PathCollection)
            ):
                continue

            if hasattr(artist, "my_data"):
                raw_x = artist.my_data["raw_x"]
                raw_y = artist.my_data["raw_y"]
                raw_color = artist.my_data["raw_color"]
                max_compress_pixel = artist.my_data["max_compress_pixel"]
            else:
                continue

            if hasattr(ax.figure, "my_timeslider"):
                valmin = ax.figure.my_timeslider.valmin
                valmax = ax.figure.my_timeslider.valmax
                valrange = valmax - valmin
                timeslider_min, timeslider_max = ax.figure.my_timeslider.val
                timeslider_min = (timeslider_min - valmin) / valrange
                timeslider_max = (timeslider_max - valmin) / valrange
            else:
                timeslider_min, timeslider_max = 0, 0

            log_x = True if ax.xaxis.get_scale() in ["log", "symlog"] else False
            log_y = True if ax.yaxis.get_scale() in ["log", "symlog"] else False
            if log_x:
                ax_range_x_min, ax_range_x_max = (
                    np.log10(ax.get_xlim()[0]),
                    np.log10(ax.get_xlim()[1]),
                )
            else:
                ax_range_x_min, ax_range_x_max = (
                    ax.get_xlim()[0],
                    ax.get_xlim()[1],
                )
            if log_y:
                ax_range_y_min, ax_range_y_max = (
                    np.log10(ax.get_ylim()[0]),
                    np.log10(ax.get_ylim()[1]),
                )
            else:
                ax_range_y_min, ax_range_y_max = (
                    ax.get_ylim()[0],
                    ax.get_ylim()[1],
                )
            ax_range_x = ax_range_x_max - ax_range_x_min
            ax_range_y = ax_range_y_max - ax_range_y_min
            ax_width_pixel = (
                ax.get_position().width * ax.figure.get_figwidth() * ax.figure.dpi
            )
            ax_height_pixel = (
                ax.get_position().height * ax.figure.get_figheight() * ax.figure.dpi
            )

            if full_load:
                compress_param = None
            else:
                compress_param = [
                    ax_range_x_min,
                    ax_range_x_max,
                    ax_range_y_min,
                    ax_range_y_max,
                    ax_range_x,
                    ax_range_y,
                    ax_width_pixel,
                    ax_height_pixel,
                    log_x,
                    log_y,
                    timeslider_min,
                    timeslider_max,
                ]

            if check_need_update_compress(
                compress_param, artist.my_data["compress_param"]
            ):
                t0 = time.time()
                artist.my_data["compress_param"] = compress_param
                data_type = np.float32
                x = (
                    np.log10(raw_x, dtype=data_type)
                    if log_x
                    else np.array(raw_x, dtype=data_type)
                )
                y = (
                    np.log10(raw_y, dtype=data_type)
                    if log_y
                    else np.array(raw_y, dtype=data_type)
                )
                length = len(x)
                display_range_min_x = ax_range_x_min - ax_range_x * (
                    zoom_out_factor / 2 + 0.1
                )
                display_range_max_x = ax_range_x_max + ax_range_x * (
                    zoom_out_factor / 2 + 0.1
                )
                display_range_min_y = ax_range_y_min - ax_range_y * (
                    zoom_out_factor / 2 + 0.1
                )
                display_range_max_y = ax_range_y_max + ax_range_y * (
                    zoom_out_factor / 2 + 0.1
                )
                invalid_mask = (
                    (x < display_range_min_x)
                    | (x > display_range_max_x)
                    | (y < display_range_min_y)
                    | (y > display_range_max_y)
                )
                invalid_mask = trim_out_range_mask(invalid_mask)
                if timeslider_min or timeslider_max:
                    time_invalid_mask = np.ones(length, dtype=bool)
                    time_count_min = max(int(timeslider_min * length), 0)
                    time_count_max = min(int(timeslider_max * length), length)
                    time_invalid_mask[time_count_min:time_count_max] = False
                    invalid_mask |= time_invalid_mask
                if np.any(invalid_mask):
                    x[invalid_mask] = np.nan
                    y[invalid_mask] = np.nan

                valid_data_mask = np.isfinite(x) & np.isfinite(y)
                valid_length = np.count_nonzero(valid_data_mask)
                cut_off_mask = np.concatenate(
                    [[False], ~valid_data_mask[1:] & valid_data_mask[:-1]]
                )
                if (
                    valid_length <= compress_length
                    or full_load
                    or not max_compress_pixel
                ):
                    artist.my_data["full_load_state"] = True
                    compress_data_mask = valid_data_mask
                    compress_valid_length = valid_length
                    compress_pixel = 0
                else:
                    artist.my_data["full_load_state"] = False
                    compress_data_mask, compress_valid_length, compress_pixel = (
                        get_compressed_data_mask(
                            x,
                            y,
                            ax_range_x,
                            ax_range_y,
                            ax_width_pixel,
                            ax_height_pixel,
                            max_compress_pixel=max_compress_pixel,
                            valid_data_mask=valid_data_mask,
                        )
                    )
                    t1 = time.time()
                    valid_length = max(valid_length, compress_valid_length)
                    compress_rate = (
                        (1 - compress_valid_length / valid_length)
                        if valid_length
                        else 0
                    )
                    _debug_print(
                        f"fig({ax.figure.number}), ax({id(ax) % 10000:04d}), artist({id(artist) % 10000:04d}) 像素分辨率：{compress_pixel}, 压缩耗时：{(t1 - t0) * 1e3:g}ms, 压缩率：{compress_rate * 100:.1f}% ({valid_length} -> {compress_valid_length})",
                    )
                compress_data_mask |= cut_off_mask
                raw_x = np.array(raw_x)
                if np.any(cut_off_mask):
                    raw_x[cut_off_mask] = np.nan
                if not np.array_equal(
                    compress_data_mask, artist.my_data["compress_data_mask"]
                ):
                    artist.my_data["compress_data_mask"] = compress_data_mask
                    if compress_data_mask is None:
                        compressed_x = raw_x
                        compressed_y = raw_y
                        if len(raw_color):
                            compressed_color = raw_color
                    else:
                        compressed_x = raw_x[compress_data_mask]
                        compressed_y = raw_y[compress_data_mask]
                        if len(raw_color):
                            compressed_color = raw_color[compress_data_mask]
                    if isinstance(artist, Line2D):  #  from plot()
                        artist.set_xdata(compressed_x)
                        artist.set_ydata(compressed_y)
                    else:  #  from scatter()
                        points_stack = np.column_stack([compressed_x, compressed_y])
                        artist.set_offsets(points_stack)
                        if len(raw_color):
                            artist.set_array(compressed_color)

            if not artist.my_data["full_load_state"]:
                ax_full_load_state = False
    return ax_full_load_state


def _call_back_before_draw(fig):
    t0 = time.time()
    if fig.number in plt.get_fignums():
        for ax in fig.axes:
            if hasattr(ax, "my_data"):
                full_load = ax.my_data["full_load"]
                ax.my_data["full_load"] = False
                ax_full_load_state = _update_compress_data(ax, full_load=full_load)
                if ax_full_load_state:
                    if ax.my_data["full_load_button"].get_visible():
                        ax.my_data["full_load_button"].set_visible(False)
                else:
                    if not ax.my_data["full_load_button"].get_visible():
                        ax.my_data["full_load_button"].set_visible(True)
    t1 = time.time()
    _debug_print(f"fig({fig.number}) _call_back_before_draw(): {(t1 - t0) * 1e3:g} ms")


def _hide_toolbar(fig):
    try:
        toolbar = fig.canvas.toolbar
        try:  # for Qt
            toolbar.setVisible(False)
        except:  # for TkAgg
            toolbar.pack_forget()
    except:
        pass


def plot(
    *args,
    fig_num: int | None = None,
    new_fig: bool = True,
    title: str = "",
    subtitle: str = "",
    data_name: str = "",
    x_name: str = "",
    y_name: str = "",
    color_name: str = "",
    color_bar: bool = True,
    color_map: str | Colormap | list | tuple = "coolwarm",
    color_min: float | int | None = None,
    color_max: float | int | None = None,
    color: list | tuple | np.ndarray | None = None,
    share_x: Axes | None = None,
    share_y: Axes | None = None,
    x_limit: list | tuple | np.ndarray | None = None,
    y_limit: list | tuple | np.ndarray | None = None,
    equal_scale: bool = False,
    dpi: float | int = 100,
    alpha: float | int = 0.7,
    log_x: bool = False,
    log_y: bool = False,
    line_width: int | None = None,
    line_color: str | tuple | list | float | None = None,
    line_style: str = "-",
    line_marker: str = ".",
    line_marker_size: float | int = 5,
    mask: list | tuple | np.ndarray | None = None,
    max_compress_pixel: float | int = 20,
    time_range: list | tuple | np.ndarray | None = None,
    data_info: dict | None = None,
    **kwargs,
):
    """
    Creates and customizes a 2D plot using Matplotlib.

    This function provides a high-level interface for generating plots with extensive
    customization options for appearance, interactivity, and data handling. It supports
    both simple line/scatter plots and complex colored scatter plots with interactive
    colorbars and sliders. The function also includes built-in event handling for
    common user interactions like panning, zooming, and data point selection.

    Args:
        *args: Variable length argument list.
               If two arguments are provided (X, Y), they are treated as the x and y
               coordinates for the plot. If one argument is provided (Y), it is plotted
               against its index range.
        fig_num (int, optional): The number of the figure to use. If `None` and `new_fig`
                                 is True, a new figure with an auto-generated number is created.
                                 If `None` and `new_fig` is False, the current figure is used.
                                 Defaults to `None`.
        new_fig (bool, optional): If True, creates a new figure window. If False, attempts
                                  to draw on the existing current figure (`plt.gcf()`).
                                  Defaults to `True`.
        title (str, optional): The main title for the entire figure window.
                               Defaults to an empty string `""`.
        subtitle (str, optional): A secondary title placed above the plotting area.
                                  Defaults to an empty string `""`.
        data_name (str, optional): The name of the dataset, used as the label in the plot's legend.
                                   Defaults to an empty string `""`.
        x_name (str, optional): Label for the x-axis.
                                Defaults to an empty string `""`.
        y_name (str, optional): Label for the y-axis.
                                Defaults to an empty string `""`.
        color_name (str, optional): Label for the colorbar, if one is displayed.
                                    Defaults to an empty string `""`.
        color_bar (bool, optional): Whether to display a colorbar for the plot when a `color`
                                    array is provided. Defaults to `True`.
        color_map (str, Colormap, list, tuple, optional): The colormap to use for mapping scalar data
                                                          to colors in scatter plots. Can be a named
                                                          colormap string, a `Colormap` object,
                                                          or a list/tuple of colors.
                                                          Defaults to `"coolwarm"`.
        color_min (float, int, None, optional): The minimum value for the color scale. If `None`,
                                                it will be inferred from the `color` data.
                                                Defaults to `None`.
        color_max (float, int, None, optional): The maximum value for the color scale. If `None`,
                                                it will be inferred from the `color` data.
                                                Defaults to `None`.
        color (list, tuple, np.ndarray, None, optional): An array of values used to map colors
                                                         to each point in the scatter plot. If
                                                         provided, a colored scatter plot is created.
                                                         Defaults to `None`.
        share_x (Axes, None, optional): An existing matplotlib `Axes` object whose x-axis this
                                        plot's x-axis will share, enforcing synchronized limits.
                                        Defaults to `None`.
        share_y (Axes, None, optional): An existing matplotlib `Axes` object whose y-axis this
                                        plot's y-axis will share, enforcing synchronized limits.
                                        Defaults to `None`.
        x_limit (list, tuple, np.ndarray, None, optional): A sequence `[min, max]` defining the
                                                           fixed limits for the x-axis. e.g., `(0, 10)`.
                                                           Defaults to `None`.
        y_limit (list, tuple, np.ndarray, None, optional): A sequence `[min, max]` defining the
                                                           fixed limits for the y-axis. e.g., `(0, 100)`.
                                                           Defaults to `None`.
        equal_scale (bool, optional): If True, sets the aspect ratio of the x and y axes to be equal,
                                      ensuring that circles appear circular. Defaults to `False`.
        dpi (float, int, optional): The resolution in dots per inch for the figure.
                                    Defaults to `100`.
        alpha (float, int, optional): The transparency level for plot elements, where 0 is fully
                                      transparent and 1 is fully opaque. Defaults to `0.7`.
        log_x (bool, optional): If True, sets the x-axis to a logarithmic scale.
                                Defaults to `False`.
        log_y (bool, optional): If True, sets the y-axis to a logarithmic scale.
                                Defaults to `False`.
        line_width (int, None, optional): The width of the lines in line plots. If `None`, the default
                                          width is used. Defaults to `None`.
        line_color (str, tuple, list, float, None, optional): The color of the lines and markers.
                                                              Can be a string (e.g., 'red', '#FF0000'),
                                                              a tuple/list of RGB(A) values, or a single
                                                              float for grayscale. If `None`, a default
                                                              color cycle is used. Defaults to `None`.
        line_style (str, optional): The style of the line, e.g., `'-'` for solid, `'--'` for dashed,
                                    `'-.'` for dash-dot, `':'` for dotted. Defaults to `'-'`.
        line_marker (str, optional): The marker style for data points, e.g., `'.'` for point,
                                     `'o'` for circle, `'^'` for triangle. Defaults to `'.'`.
        line_marker_size (float, int, optional): The size of the markers. Must be greater than 0.
                                                 Defaults to `5`.
        mask (list, tuple, np.ndarray, None, optional): A boolean array of the same length as the data.
                                                        Points corresponding to `False` values in the mask
                                                        will be hidden (set to NaN) on the plot.
                                                        Defaults to `None`.
        max_compress_pixel (float, int, optional): A threshold for enabling data compression for large
                                                   datasets. If the number of data points exceeds the
                                                   estimated pixel density defined by this value, the plot
                                                   may be optimized for performance by simplifying the
                                                   visual representation. Setting to 0 disables compression.
                                                   Defaults to `20`.
        time_range (list, tuple, np.ndarray, None, optional): A sequence `[start, end]` defining the
                                                              temporal bounds for an interactive time slider.
                                                              This is typically used when `max_compress_pixel`
                                                              is active and the data is time-series-like.
                                                              Defaults to `None`.
        data_info (dict, None, optional): A dictionary containing additional metadata associated with
                                          each data point. Keys can be arbitrary strings, and values must
                                          be arrays of the same length as the input data (x/y). This data
                                          can be used for tooltips or advanced interactivity.
                                          Defaults to `None`.

    Returns:
        matplotlib.axes.Axes: The Axes object on which the plot was drawn, allowing for further
                              customization if needed.
    """

    # Warn about unexpected parameters
    if kwargs:
        unexpected = ", ".join(f"'{k}'" for k in kwargs.keys())
        warnings.warn(
            f"plot() received unexpected parameter(s): {unexpected}. These will be ignored.",
            UserWarning,
            stacklevel=2,
        )

    plt.rcParams["font.family"] = "Microsoft YaHei"
    # plt.rcParams.update({"font.size":5})
    plt.rcParams.update({"figure.max_open_warning": 0})

    # init x, y, length
    if len(args) == 2:
        x = np.array(args[0], dtype=np.float64)
        y = np.array(args[1], dtype=np.float64)
        x = np.where(np.isfinite(x), x, np.nan)
        y = np.where(np.isfinite(y), y, np.nan)
    elif len(args) == 1:
        x = np.arange(len(args[0]), dtype=np.float64)
        y = np.array(args[0], dtype=np.float64)
        y = np.where(np.isfinite(y), y, np.nan)
    else:
        raise TypeError(
            f"plot() takes 1 or 2 positional arguments but {len(args)} were given"
        )
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        raise ValueError(f"len(x) = {len(x)}, len(y) = {len(y)}")
    length = min(len(x), len(y))
    # check fig_num
    if not isinstance(fig_num, (int, NoneType)):
        raise TypeError(
            f"fig_num: expected int | None, but got {type(fig_num).__name__}"
        )
    # check new_fig
    if not isinstance(new_fig, (bool)):
        raise TypeError(f"new_fig: expected bool, but got {type(new_fig).__name__}")
    # check title
    if not isinstance(title, (str)):
        raise TypeError(f"title: expected str, but got {type(title).__name__}")
    # check subtitle
    if not isinstance(subtitle, (str)):
        raise TypeError(f"subtitle: expected str, but got {type(subtitle).__name__}")
    # check data_name
    if not isinstance(data_name, (str)):
        raise TypeError(f"data_name: expected str, but got {type(data_name).__name__}")
    # check x_name
    if not isinstance(x_name, (str)):
        raise TypeError(f"x_name: expected str, but got {type(x_name).__name__}")
    # check y_name
    if not isinstance(y_name, (str)):
        raise TypeError(f"y_name: expected str, but got {type(y_name).__name__}")
    # check color_name
    if not isinstance(color_name, (str)):
        raise TypeError(
            f"color_name: expected str, but got {type(color_name).__name__}"
        )
    # check color_bar
    if not isinstance(color_bar, (bool)):
        raise TypeError(f"color_bar: expected bool, but got {type(color_bar).__name__}")
    # check color_map
    if not isinstance(color_map, (str, Colormap, list, tuple)):
        raise TypeError(
            f"color_map: expected str | Colormap | list | tuple, but got {type(color_map).__name__}"
        )
    # check color_min
    if not isinstance(color_min, (float, int, NoneType)):
        raise TypeError(
            f"color_min: expected float | int | None, but got {type(color_min).__name__}"
        )
    # check color_max
    if not isinstance(color_max, (float, int, NoneType)):
        raise TypeError(
            f"color_max: expected float | int | None, but got {type(color_max).__name__}"
        )
    # check color
    if not isinstance(color, (list, tuple, np.ndarray, NoneType)):
        raise TypeError(
            f"color: expected list | tuple | np.ndarray | None, but got {type(color).__name__}"
        )
    if color is not None and len(color) != length:
        raise ValueError(f"len(color) = {len(color)} != {length}")
    # check share_x
    if not isinstance(share_x, (Axes, NoneType)):
        raise TypeError(
            f"share_x: expected Axes | None, but got {type(share_x).__name__}"
        )
    # check share_y
    if not isinstance(share_y, (Axes, NoneType)):
        raise TypeError(
            f"share_y: expected Axes | None, but got {type(share_y).__name__}"
        )
    # check x_limit
    if not isinstance(x_limit, (list, tuple, np.ndarray, NoneType)):
        raise TypeError(
            f"x_limit: expected list | tuple | np.ndarray | None, but got {type(x_limit).__name__}"
        )
    if x_limit is not None and len(x_limit) != 2:
        raise ValueError(f"len(x_limit) = {len(x_limit)} != {2}")
    # check y_limit
    if not isinstance(y_limit, (list, tuple, np.ndarray, NoneType)):
        raise TypeError(
            f"y_limit: expected list | tuple | np.ndarray | None, but got {type(y_limit).__name__}"
        )
    if y_limit is not None and len(y_limit) != 2:
        raise ValueError(f"len(y_limit) = {len(y_limit)} != {2}")
    # check equal_scale
    if not isinstance(equal_scale, (bool)):
        raise TypeError(
            f"equal_scale: expected bool, but got {type(equal_scale).__name__}"
        )
    # check dpi
    if not isinstance(dpi, (float, int)):
        raise TypeError(f"dpi: expected float | int, but got {type(dpi).__name__}")
    if dpi is not None and dpi <= 0:
        raise ValueError(f"dpi <= 0")
    # check alpha
    if not isinstance(alpha, (float, int)):
        raise TypeError(f"alpha: expected float | int, but got {type(alpha).__name__}")
    # check log_x
    if not isinstance(log_x, (bool)):
        raise TypeError(f"log_x: expected bool, but got {type(log_x).__name__}")
    # check log_y
    if not isinstance(log_y, (bool)):
        raise TypeError(f"log_y: expected bool, but got {type(log_y).__name__}")
    # check line_width
    if not isinstance(line_width, (int, NoneType)):
        raise TypeError(
            f"line_width: expected int | None, but got {type(line_width).__name__}"
        )
    # check line_color
    if not isinstance(line_color, (str, tuple, list, float, NoneType)):
        raise TypeError(
            f"line_color: expected str | tuple | list | float | None, but got {type(line_color).__name__}"
        )
    # check line_style
    if not isinstance(line_style, (str)):
        raise TypeError(
            f"line_style: expected str, but got {type(line_style).__name__}"
        )
    # check line_marker
    if not isinstance(line_marker, (str)):
        raise TypeError(
            f"line_marker: expected str, but got {type(line_marker).__name__}"
        )
    # check line_marker_size
    if not isinstance(line_marker_size, (float, int)):
        raise TypeError(
            f"line_marker_size: expected float | int, but got {type(line_marker_size).__name__}"
        )
    if line_marker_size <= 0:
        raise ValueError(f"line_marker_size <= 0")
    # check mask
    if not isinstance(mask, (list, tuple, np.ndarray, NoneType)):
        raise TypeError(
            f"mask: expected list | tuple | np.ndarray | None, but got {type(mask).__name__}"
        )
    # check max_compress_pixel
    if not isinstance(max_compress_pixel, (float, int)):
        raise TypeError(
            f"max_compress_pixel: expected float | int, but got {type(max_compress_pixel).__name__}"
        )
    if max_compress_pixel < 0:
        raise ValueError(f"max_compress_pixel < 0")
    # check time_range
    if not isinstance(time_range, (list, tuple, np.ndarray, NoneType)):
        raise TypeError(
            f"time_range: expected NoneType | list | tuple | np.ndarray, but got {type(time_range).__name__}"
        )
    if time_range is not None:
        if not (len(time_range) == 2 and time_range[0] < time_range[1]):
            raise ValueError(
                "time_range: must be a sequence of 2 elements with start < end"
            )
    # check data_info
    if not isinstance(data_info, (dict, NoneType)):
        raise TypeError(
            f"data_info: expected dict | None, but got {type(data_info).__name__}"
        )
    if data_info is not None:
        for key in data_info:
            if len(data_info[key]) != length:
                raise ValueError(
                    f'len(data_info["{key}"]) = {len(data_info[key])} != {length}'
                )
    # init color, color_min, color_max
    if color is None:
        color = []
    color = np.array(color)
    color_min = (
        min(color)
        if len(color) and (color_min is None or color_min == -np.inf)
        else color_min
    )
    color_max = (
        max(color)
        if len(color) and (color_max is None or color_max == np.inf)
        else color_max
    )
    # init mask
    if mask is None:
        mask = []
    mask = np.array(mask, dtype=bool)
    # init fig, new_fig
    if fig_num is None and new_fig:
        fig = plt.figure(dpi=dpi)
    else:
        if fig_num is None:
            fig_nums = plt.get_fignums()
            fig_num = max(fig_nums) if fig_nums else 1
        if plt.fignum_exists(fig_num):
            if new_fig:
                plt.close(fig_num)
                fig = plt.figure(dpi=dpi)
            else:
                fig = plt.figure(num=fig_num)
        else:
            fig = plt.figure(num=fig_num, dpi=dpi)
            new_fig = True
    # init ax
    if new_fig:
        fig.clf()
        ax = fig.add_subplot(1, 1, 1, sharex=share_x, sharey=share_y)
    else:
        axes = fig.get_axes()
        assert len(axes), "there are no figure, please use new_fig=True !"
        ax = axes[0]
    if title:
        fig.suptitle(title, fontsize=15)
        try:
            fig.canvas.manager.window.setWindowTitle(f"Figure {fig.number}: {title}")
        except:
            pass
    if subtitle:
        ax.set_title(subtitle, fontsize=10)
    # init x, y, color, mask, empty_x, empty_y, empty_color
    x = x[:length]
    y = y[:length]
    if len(color):
        color = color[:length]
    if len(mask) >= length:
        mask = mask[:length]
    else:
        mask = []
    if len(mask):
        x = np.copy(x)
        y = np.copy(y)
        x[mask == False] = np.nan
        y[mask == False] = np.nan
    if max_compress_pixel:
        empty_x = [min(x), np.nan, min(x), np.nan, max(x), np.nan, max(x)]
        empty_y = [min(y), np.nan, max(y), np.nan, max(y), np.nan, min(y)]
        empty_color = [0] * len(empty_x)
    # create artist, artist.my_line, fig.my_colorbar, fig.my_colorslider
    if len(color):
        if line_style and length <= 1000 and not max_compress_pixel:
            points_stack = np.column_stack([x, y])
            segments = np.stack([points_stack[:-1], points_stack[1:]], axis=1)
            line = LineCollection(
                segments,
                norm=Normalize(color_min, color_max),
                alpha=alpha,
                cmap=color_map,
                linewidths=line_width,
                linestyles=line_style,
            )
            line.set_array((color[:-1] + color[1:]) / 2)
            ax.add_collection(line)
            ax.autoscale()
        else:
            line = None
        artist = ax.scatter(
            x if not max_compress_pixel else empty_x,
            y if not max_compress_pixel else empty_y,
            c=color if not max_compress_pixel else empty_color,
            marker=line_marker,
            s=line_marker_size**2,
            label=data_name,
            alpha=alpha,
            cmap=color_map,
            vmin=color_min,
            vmax=color_max,
        )
        if line:
            artist.my_line = line
        if color_bar:
            if not hasattr(fig, "my_colorslider"):
                colorbar = plt.colorbar(artist)
                if color_name:
                    colorbar.set_label(color_name)
                cmin, cmax = artist.get_clim()
                bbox = colorbar.ax.get_position()
                colorslider_x0 = bbox.x1 + 0.03
                colorslider_y0 = bbox.y0
                colorslider_height = bbox.height
                colorslider_width = 0.013  # colorslider_height * 0.03
                colorslider_ax = fig.add_axes(
                    [
                        colorslider_x0,
                        colorslider_y0,
                        colorslider_width,
                        colorslider_height,
                    ]
                )
                colorslider = RangeSlider(
                    colorslider_ax,
                    "color",
                    cmin,
                    cmax,
                    valinit=(cmin, cmax),
                    orientation="vertical",
                )
                colorslider.poly.set_facecolor("gray")
                if line:
                    colorslider.on_changed(
                        lambda val, artist=artist, line=line: (
                            artist.set_clim(*val),
                            line.set_clim(*val),
                        )
                    )
                else:
                    colorslider.on_changed(
                        lambda val, artist=artist: artist.set_clim(*val)
                    )
                fig.my_colorbar = colorbar
                fig.my_colorslider = colorslider
            else:
                colorslider = fig.my_colorslider
                if line:
                    colorslider.on_changed(
                        lambda val, artist=artist, line=line: (
                            artist.set_clim(*val),
                            line.set_clim(*val),
                        )
                    )
                else:
                    colorslider.on_changed(
                        lambda val, artist=artist: artist.set_clim(*val)
                    )
    else:
        artists = ax.plot(
            x if not max_compress_pixel else empty_x,
            y if not max_compress_pixel else empty_y,
            marker=line_marker,
            markersize=line_marker_size,
            label=data_name,
            alpha=alpha,
            color=line_color,
            lw=line_width,
            linestyle=line_style,
        )
        artist = artists[-1]
    # set log_x, log_y, x_name, y_name, data_name
    if log_x:
        ax.set_xscale("symlog", linthresh=1e-10)
    if log_y:
        ax.set_yscale("symlog", linthresh=1e-10)
    if x_name:
        ax.set_xlabel(x_name)
    if y_name:
        ax.set_ylabel(y_name)
    if data_name:
        legend = ax.legend(loc="upper right")
        legend.set_picker(True)
    # create fig.my_timeslider
    if (
        max_compress_pixel
        and not hasattr(fig, "my_timeslider")
        and np.any(x[1:] <= x[:1])
    ):
        bbox = ax.get_position()
        tightbbox = ax.get_tightbbox(fig.canvas.get_renderer())
        tightbbox = tightbbox.transformed(fig.transFigure.inverted())
        timeslider_width = bbox.width
        timeslider_height = 0.0155  #  timeslider_width * 0.025
        timeslider_x0 = bbox.x0
        timeslider_y0 = tightbbox.y0 / 3 * 2
        timeslider_ax = fig.add_axes(
            [timeslider_x0, timeslider_y0, timeslider_width, timeslider_height]
        )
        if time_range is None:
            timeslider = RangeSlider(
                timeslider_ax,
                "time  ",
                0,
                1,
                valinit=(0, 1),
                orientation="horizontal",
            )
            timeslider.valtext.set_visible(False)
            timeslider.poly.set_facecolor("gray")
            # timeslider.on_changed(lambda val: ())
            fig.my_timeslider = timeslider
        else:
            time_min = time_range[0]
            time_max = time_range[1]
            timeslider = RangeSlider(
                timeslider_ax,
                "time  ",
                time_min,
                time_max,
                valinit=(time_min, time_max),
                orientation="horizontal",
            )
            timeslider.poly.set_facecolor("gray")
            # timeslider.on_changed(lambda val: ())
            fig.my_timeslider = timeslider
    # create artist.my_data
    artist.my_data = {
        "raw_x": x,
        "raw_y": y,
        "raw_color": color,
        "compress_param": None,
        "compress_data_mask": None,
        "full_load_state": False,
        "max_compress_pixel": max_compress_pixel,
        "data_info": data_info,
    }
    # create fig.my_data
    if not hasattr(fig, "my_data"):
        fig.my_data = {
            "current_ax": None,
        }
    # create ax.my_data
    if not hasattr(ax, "my_data"):
        ax.my_data = {
            # ax
            "ax_range": None,
            "ax_background": None,
            "equal_scale": None,
            # button
            "button_release_time": None,
            "left_button_pressed_time": None,
            "right_button_pressed_time": None,
            "right_button_timer": None,
            "mouse_button_hold_time_threshold": 0.3,
            # cursor
            "cursor_select_is_busy": False,
            "cursor_drag_is_busy": False,
            "picked_cursor": None,
            # legend
            "picked_legend": None,
            # measure
            "measure_is_busy": False,
            "measure_start_point": None,
            "measure_patch": [],
            "measure_text_d": [],
            "measure_text_dx": [],
            "measure_text_dy": [],
            # rectangle select
            "rectangle_select_is_busy": False,
            "rectangle_select_start_point": None,
            "rectangle_select": None,
            # help
            "help_text": None,
            "help_button": None,
            # compress
            "full_load": False,
        }
    # update ax.my_data
    ax.my_data["equal_scale"] = equal_scale
    # set x_limit, y_limit, equal_scale
    _auto_scale(ax)
    if x_limit:
        ax.set_xlim(x_limit[0], x_limit[1])
    if y_limit:
        ax.set_ylim(y_limit[0], y_limit[1])
    ax.grid("on")
    if equal_scale:
        ax.set_aspect("equal", adjustable="box")
    # reset fig.canvas.manager.toolbar._nav_stack
    _push_ax(ax)
    try:
        if not new_fig:
            _nav_stack = fig.canvas.manager.toolbar._nav_stack
            if _nav_stack._pos > 0:
                # update stack[0] during drawing to ensure the correct home of figure
                _nav_stack._elements[0] = _nav_stack._elements[_nav_stack._pos]
    except:
        pass
    # create help_text, help_button, full_load_button
    if new_fig:
        # init help_info
        if _check_language_Chinese() == True:
            help_info = "\n".join(
                (
                    "【操作说明】",
                    "• 框选视图:  左键拖拽",
                    "• 重置视图:  右键单击",
                    "• 平移视图:  右键拖拽",
                    "• 缩放视图:  中键滚动",
                    "• Y轴自适应:  右键长按",
                    "• 显示数据信息:  左键单击 (数据点)",
                    "• 隐藏数据信息:  右键单击 (数据信息)",
                    "• 开关曲线显示:  左键单击 (图例)",
                    "• 标尺工具:  中键拖拽",
                    "• 清除标尺:  中键单击",
                    "• 历史视图后退:  键盘 ← ",
                    "• 历史视图前进:  键盘 → ",
                    "• 重载无压缩视图:  左键点击" + r"$\boldsymbol{\circlearrowright}$",
                )
            )
        else:
            help_info = "\n".join(
                (
                    "【Operation Instructions】",
                    "• Zoom Into:  Left Drag",
                    "• Reset View:  Right Click",
                    "• Pan:  Right Drag",
                    "• Zoom Scroll:  Middle Wheel",
                    "• Auto-Scale Y:  Right Long Press",
                    "• Show Data Info:  Left Click (on point)",
                    "• Hide Data Info:  Right Click (on data info)",
                    "• Toggle Curve:  Left Click (on legend)",
                    "• Ruler Tool:  Middle Drag",
                    "• Clear Ruler:  Middle Click",
                    "• History View Back:  Keyboard ←",
                    "• History View Forward:  Keyboard →",
                    "• Reload Uncompressed View:  Left Click "
                    + r"$\boldsymbol{\circlearrowright}$",
                )
            )
        ax.my_data["help_text"] = ax.text(
            -0.05,
            1.05,
            help_info,
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round, pad=1",
                facecolor=(0.5, 0.5, 0.5, 0.95),
                edgecolor="none",
            ),
            visible=False,
        )
        ax.my_data["help_button"] = ax.text(
            -0.12,
            1.12,
            r"$\boldsymbol{?}$",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color="white",
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="circle, pad=0.3",
                facecolor=(0.5, 0.5, 0.5, 0.6),
                edgecolor="none",
            ),
            picker=True,
        )
        ax.my_data["full_load_button"] = ax.text(
            -0.12,
            -0.03,
            r"$\boldsymbol{\circlearrowright}$",
            fontfamily="sans-serif",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color="white",
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="circle, pad=0.3",
                facecolor=(0.5, 0.5, 0.5, 0.6),
                edgecolor="none",
            ),
            picker=True,
            visible=False,
        )
    # create cursor
    if "cursor" in ax.my_data:
        ax.my_data["cursor"].remove()
    artists = [
        artist
        for artist in ax.get_children()
        if isinstance(artist, (Line2D, PathCollection))
    ]
    ax.my_data["cursor"] = mplcursors.cursor(
        artists,
        multiple=True,
        bindings={"select": -1, "deselect": -2},
        hover=False,
    )
    for disconnectors in ax.my_data["cursor"]._disconnectors:
        disconnectors()
    ax.my_data["cursor"].connect("add", _call_back_on_add_cursor)
    # init toolbar, create callback, create timer, save draw_original
    if new_fig:
        # init toolbar
        _hide_toolbar(fig)
        # create callback
        ax.figure.canvas.mpl_connect("button_press_event", _call_back_on_button_press)
        ax.figure.canvas.mpl_connect(
            "button_release_event", _call_back_on_button_release
        )
        ax.figure.canvas.mpl_connect("scroll_event", _call_back_on_scroll)
        ax.figure.canvas.mpl_connect("pick_event", _call_back_on_pick)
        ax.figure.canvas.mpl_connect("motion_notify_event", _call_back_on_motion)
        # create right_button_timer
        ax.my_data["right_button_timer"] = ax.figure.canvas.new_timer(
            interval=ax.my_data["mouse_button_hold_time_threshold"] * 1000
        )
        ax.my_data["right_button_timer"].single_shot = True
        ax.my_data["right_button_timer"].add_callback(
            _call_back_on_right_button_timeout
        )
        # save original fig.canvas.draw
        fig.canvas.draw_original = fig.canvas.draw

    # create callback before fig.canvas.draw
    def _draw_with_call_back(*args, **kwargs):
        _call_back_before_draw(fig)
        return fig.canvas.draw_original(*args, **kwargs)

    # overwrite fig.canvas.draw
    if max_compress_pixel:
        fig.canvas.draw = _draw_with_call_back

    """
    plt.ion()
    plt.draw()
    plt.pause(0.001)
    plt.ioff()
    """
    return ax


def show_figure():
    plt.show()


def close_figure(fig_num: int | None = None):
    """
    Args:
        fig_num (int, optional): The number of the figure to close. If None, closes all figures.
    """
    if not isinstance(fig_num, (int, NoneType)):
        raise TypeError(
            f"fig_num: expected int | None, but got {type(fig_num).__name__}"
        )
    if fig_num is None:
        plt.close("all")
        _debug_print(f"closes all figures!")
    else:
        plt.close(fig_num)
        _debug_print(f"closes figures {fig_num}!")


def welch(
    x: list | tuple | np.ndarray,
    y: list | tuple | np.ndarray | None = None,
    fs: int | float | np.number = 1.0,
    nperseg: int | np.integer | None = None,
    noverlap: int | np.integer | None = None,
    seg_padded: bool = False,
    nfft: int | np.integer | None = None,
    window: str | list | tuple | np.ndarray = "hann",
    detrend: str = "constant",
    mode: str = "amplitude",
    **kwargs,
):
    """
    Welch's method for spectral analysis.

    Computes the power spectral density (PSD) or other spectral quantities
    using Welch's method of averaged periodograms.

    Parameters
    ----------
    x : array_like
        Input signal (time-domain data). Must be a 1-D array with length > 2.
    y : array_like, optional
        Output signal for transfer function estimation. If provided, must have
        the same length as x. Default is None.
    fs : float, optional
        Sampling frequency in Hz. Must be positive and greater than machine epsilon.
        Default is 1.0.
    nperseg : int or None, optional
        Length of each segment. If None, set to len(x) (no segmentation).
        Must be at least 2 and cannot exceed the length of x. Default is None.
    noverlap : int or None, optional
        Number of points to overlap between segments. If None, set to
        ceil(nperseg / 2). Must be less than nperseg. Default is None.
    seg_padded : bool, optional
        If True, pad x with zeros to make its length an integer multiple of
        the step size (nperseg - noverlap). If False, truncate excess data.
        Default is False.
    nfft : int or None, optional
        Length of the FFT used. If None, set to nperseg. Must be >= nperseg.
        If nfft > nperseg, zero-padding is applied to increase frequency resolution.
        Default is None.
    window : str, array_like, or None, optional
        Window function to apply. Options:
        - "hann": Hann window (default)
        - "hamm": Hamming window
        - "black": Blackman window
        - "flat": Flat-top window
        - None: Rectangular window (all ones)
        - array_like: Custom window coefficients (length must equal nperseg)
        Default is "hann".
    detrend : str, optional
        Detrending method applied to each segment before windowing:
        - "off": No detrending
        - "constant": Remove mean (default)
        - "linear": Remove linear trend
        Default is "constant".
    mode : str, optional
        Type of spectral computation:
        - "complex": Raw complex FFT results
        - "amplitude": Amplitude spectrum (magnitude and phase)
        - "power": Power spectrum
        - "psd": Power spectral density (normalized by sampling rate)
        - "response": Frequency response function (requires y input)
        Default is "amplitude".

    Returns
    -------
    result : tuple or ndarray
        The return value depends on the mode:

        - mode="complex": ndarray, shape (n_segments, nfft)
            Raw complex FFT results for each segment.

        - mode="amplitude": tuple of (freq, amp, phase)
            - freq: ndarray of frequencies (Hz), shape (n_freq,)
            - amp: ndarray of amplitude values, shape (n_freq,)
            - phase: ndarray of phase values (radians), shape (n_freq,)

        - mode="power": tuple of (freq, power)
            - freq: ndarray of frequencies (Hz), shape (n_freq,)
            - power: ndarray of power spectrum values, shape (n_freq,)

        - mode="psd": tuple of (freq, psd)
            - freq: ndarray of frequencies (Hz), shape (n_freq,)
            - psd: ndarray of power spectral density values, shape (n_freq,)

        - mode="response": tuple of (freq, gain, phase, coherence)
            - freq: ndarray of frequencies (Hz), shape (n_freq,)
            - gain: ndarray of magnitude response, shape (n_freq,)
            - phase: ndarray of phase response (radians), shape (n_freq,)
            - coherence: ndarray of magnitude-squared coherence, shape (n_freq,)

    Raises
    ------
    TypeError
        If input types are invalid (e.g., non-numeric x, non-boolean seg_padded).
    ValueError
        If parameter values are invalid (e.g., fs <= 0, nperseg < 2,
        noverlap >= nperseg, nfft < nperseg, window size mismatch,
        unsupported window type, invalid detrend option, invalid mode,
        or y is None when mode="response").

    Examples
    --------
    >>> import numpy as np
    >>> fs = 1000.0  # Sampling frequency
    >>> t = np.arange(0, 1, 1/fs)
    >>> x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(len(t))
    >>> freq, amp, phase = welch(x, fs=fs, nperseg=256, mode="amplitude")

    Notes
    -----
    - For real signals, only the positive frequency components are returned
      (n_freq = nfft // 2 + 1).
    - The DC component (freq=0) and Nyquist frequency (if nfft is even) are
      not doubled; all other components are doubled to account for the
      one-sided spectrum.
    - Coherence values are clipped to the range [0, 1].
    """

    # Warn about unexpected parameters
    if kwargs:
        unexpected = ", ".join(f"'{k}'" for k in kwargs.keys())
        warnings.warn(
            f"welch() received unexpected parameter(s): {unexpected}. These will be ignored.",
            UserWarning,
            stacklevel=2,
        )

    def get_window(window, n):
        """Generate window coefficients."""
        if window is None:
            return np.ones(n)
        elif isinstance(window, str):
            k = np.arange(n)
            if window.lower() == "hann":
                return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)
            elif window.lower() == "hamm":
                return 0.54 - 0.46 * np.cos(2.0 * np.pi * k / n)
            elif window.lower() == "black":  # Blackman
                return (
                    0.42
                    - 0.5 * np.cos(2.0 * np.pi * k / n)
                    + 0.08 * np.cos(4.0 * np.pi * k / n)
                )
            elif window.lower() == "flat":  # Flattop
                return (
                    0.21557895
                    - 0.41663158 * np.cos(2.0 * np.pi * k / n)
                    + 0.277263158 * np.cos(4.0 * np.pi * k / n)
                    - 0.083578947 * np.cos(6.0 * np.pi * k / n)
                    + 0.006947368 * np.cos(8.0 * np.pi * k / n)
                )
            else:
                raise ValueError(
                    f"Unsupported window type: '{window}'. "
                    f"Supported options: 'hann', 'hamm', 'black', 'flat', or None."
                )
        elif isinstance(window, (list, tuple, np.ndarray)):
            window_arr = np.asarray(window)
            if window_arr.ndim != 1:
                raise ValueError(
                    f"Window must be 1-D array, got shape {window_arr.shape}"
                )
            if len(window_arr) != n:
                raise ValueError(
                    f"Window length ({len(window_arr)}) must equal nperseg ({n})"
                )
            return window_arr
        else:
            raise TypeError(
                f"Window must be a string, array_like, or None, "
                f"got type {type(window).__name__}"
            )

    def detrend_linear(x):
        """Remove linear trend from data."""
        x = np.asarray(x)
        n = x.size
        if n <= 1:
            return x - x.mean()
        t = np.arange(n, dtype=x.dtype)
        a = np.dot(t, x - x.mean()) * 12.0 / (n * (n * n - 1.0))
        b = x.mean() - a * (n - 1.0) / 2.0
        return x - (a * t + b)

    # ============ Input Type Checking ============
    # Check x type
    if not isinstance(x, (list, tuple, np.ndarray)):
        raise TypeError(f"x must be array_like, got type {type(x).__name__}")
    x = np.asarray(x)
    if x.ndim == 0:
        raise ValueError("x cannot be a scalar")
    if x.ndim > 1:
        raise ValueError(f"x must be 1-D array, got shape {x.shape}")
    # Check y type if provided
    if y is not None:
        if not isinstance(y, (list, tuple, np.ndarray)):
            raise TypeError(
                f"y must be array_like or None, got type {type(y).__name__}"
            )
        y = np.asarray(y)
        if y.ndim == 0:
            raise ValueError("y cannot be a scalar")
        if y.ndim > 1:
            raise ValueError(f"y must be 1-D array, got shape {y.shape}")
    # Check fs type and value
    if not isinstance(fs, (int, float, np.number)):
        raise TypeError(f"fs must be numeric, got type {type(fs).__name__}")
    fs = float(fs)
    # Check seg_padded type
    if not isinstance(seg_padded, bool):
        raise TypeError(
            f"seg_padded must be boolean, got type {type(seg_padded).__name__}"
        )
    # Check detrend type
    if not isinstance(detrend, str):
        raise TypeError(
            f"detrend must be 'off', 'constant' or 'linear', "
            f"got type {type(detrend).__name__}"
        )
    if detrend.lower() not in ("off", "constant", "linear"):
        raise ValueError(
            f"detrend must be 'off', 'constant' or 'linear', got '{detrend}'"
        )
    # Check mode type
    if not isinstance(mode, str):
        raise TypeError(f"mode must be a string, got type {type(mode).__name__}")
    mode = mode.lower()
    valid_modes = ("complex", "amplitude", "power", "psd", "response")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
    # ============ Value Checking ============
    eps = np.finfo(float).eps
    # Check signal length
    if len(x) <= 2:
        raise ValueError(f"Length of x ({len(x)}) must be greater than 2")
    # Check sampling frequency
    if fs <= eps:
        raise ValueError(
            f"Sampling frequency fs ({fs}) must be greater than machine epsilon ({eps})"
        )
    # Reshape to 1-D
    x = x.reshape(-1)
    n = len(x)
    if y is not None:
        y = y.reshape(-1)
        if len(y) != n:
            raise ValueError(f"Length of y ({len(y)}) must equal length of x ({n})")
    # Check nperseg
    if nperseg is None:
        nperseg = n
    else:
        if not isinstance(nperseg, (int, np.integer)):
            raise TypeError(
                f"nperseg must be an integer or None, got type {type(nperseg).__name__}"
            )
        nperseg = int(nperseg)
        if nperseg < 2:
            raise ValueError(f"nperseg ({nperseg}) must be at least 2")
        if nperseg > n:
            raise ValueError(f"nperseg ({nperseg}) cannot exceed signal length ({n})")
    # Check nfft
    if nfft is None:
        nfft = nperseg
    else:
        if not isinstance(nfft, (int, np.integer)):
            raise TypeError(
                f"nfft must be an integer or None, got type {type(nfft).__name__}"
            )
        nfft = int(nfft)
        if nfft < nperseg:
            raise ValueError(
                f"nfft ({nfft}) must be greater than or equal to nperseg ({nperseg})"
            )
    # Check noverlap
    if noverlap is None:
        noverlap = int(np.ceil(nperseg / 2))
    else:
        if not isinstance(noverlap, (int, np.integer)):
            raise TypeError(
                f"noverlap must be an integer or None, got type {type(noverlap).__name__}"
            )
        if noverlap >= nperseg:
            raise ValueError(
                f"noverlap ({noverlap}) must be less than nperseg ({nperseg})"
            )
        if noverlap < 0:
            raise ValueError(f"noverlap ({noverlap}) must be non-negative")
    # Initialize nstep
    nstep = nperseg - noverlap
    # Segment padding
    if seg_padded:
        nadd = (nperseg - (n - nperseg) % nstep) % nperseg
        x = np.append(x, np.zeros(nadd))
        if y is not None:
            y = np.append(y, np.zeros(nadd))
        n = len(x)
    # Initialize nseg
    nseg = (n - noverlap) // nstep
    # Initialize FFT result arrays
    fft_result_x = np.zeros((nseg, nfft), dtype=complex)
    if y is not None:
        fft_result_y = np.zeros((nseg, nfft), dtype=complex)
    # Get window coefficients
    win = get_window(window, nperseg)
    if np.sum(win) <= eps:
        raise ValueError(
            f"Sum of window coefficients ({np.sum(win)}) is too small (<= eps)"
        )
    # Process segments
    for i in range(nseg):
        start_idx = nstep * i
        end_idx = nperseg + nstep * i
        x_seg = np.copy(x[start_idx:end_idx])
        if y is not None:
            y_seg = np.copy(y[start_idx:end_idx])
        # Detrend
        if detrend.lower() == "constant":
            x_seg = x_seg - x_seg.mean()
            if y is not None:
                y_seg = y_seg - y_seg.mean()
        elif detrend.lower() == "linear":
            x_seg = detrend_linear(x_seg)
            if y is not None:
                y_seg = detrend_linear(y_seg)
        # Apply window
        x_seg = x_seg * win
        if y is not None:
            y_seg = y_seg * win
        # Zero-pad for FFT
        if nfft > nperseg:
            nfftadd = nfft - nperseg
            x_seg = np.append(x_seg, np.zeros(nfftadd))
            if y is not None:
                y_seg = np.append(y_seg, np.zeros(nfftadd))
        # Compute FFT
        fft_result_x[i, :] = np.fft.fft(x_seg, nfft)
        if y is not None:
            fft_result_y[i, :] = np.fft.fft(y_seg, nfft)
    # Compute frequency axis
    nfreq = nfft // 2 + 1
    freq = np.arange(0, nfreq) / nfft * fs
    # ============ Mode-specific computations ============
    if mode == "complex":
        return fft_result_x
    elif mode == "amplitude":
        fft_result_x = fft_result_x[:, 0:nfreq]
        scale = 1.0 / np.sum(win)
        amp = np.abs(fft_result_x).mean(axis=0) * scale
        phase = np.angle(fft_result_x).mean(axis=0)
        # Double all frequencies except DC and Nyquist (if even)
        if nfft % 2:
            amp[1:nfreq] *= 2.0
        else:
            amp[1 : (nfreq - 1)] *= 2.0
        return freq, amp, phase
    elif mode == "power" or mode == "psd":
        fft_result_x = fft_result_x[:, 0:nfreq]
        if mode == "power":
            scale = 1.0 / np.sum(win) ** 2
        else:  # psd
            scale = 1.0 / np.sum(win**2) / fs
        Pxx = np.conj(fft_result_x) * fft_result_x * scale
        Pxx = Pxx.real.mean(axis=0)
        # Double all frequencies except DC and Nyquist (if even)
        if nfft % 2:
            Pxx[1:nfreq] *= 2.0
        else:
            Pxx[1 : (nfreq - 1)] *= 2.0
        return freq, Pxx
    elif mode == "response":
        if y is None:
            raise ValueError(
                "Mode 'response' requires both x and y inputs. y cannot be None."
            )
        fft_result_x = fft_result_x[:, 0:nfreq]
        fft_result_y = fft_result_y[:, 0:nfreq]
        # Compute cross-spectral densities
        Cxx = (np.conj(fft_result_x) * fft_result_x).mean(axis=0)
        Cxy = (np.conj(fft_result_x) * fft_result_y).mean(axis=0)
        Cyy = (np.conj(fft_result_y) * fft_result_y).mean(axis=0)
        # Avoid division by zero
        Cxx.real = np.where(Cxx.real <= eps, eps, Cxx.real)
        Cyy.real = np.where(Cyy.real <= eps, eps, Cyy.real)
        # Frequency response
        response = Cxy / Cxx
        gain = np.abs(response)
        phase = np.angle(response)
        # Magnitude-squared coherence
        coherence = np.abs(Cxy * Cxy / Cxx / Cyy)
        coherence = np.clip(coherence, 0.0, 1.0)
        return freq, gain, phase, coherence


def _test_for_single_fig():
    n = 1_000_000
    x1 = np.linspace(0, 2 * np.pi, n)
    y1 = np.sin(x1)
    x2 = np.linspace(0, 2 * np.pi, n)
    y2 = np.cos(x2)

    ax = plot(
        x1,
        y1,
        data_name="my_data_1",
        x_name="my_x",
        y_name="my_y",
        title="my_title",
        mask=y1 > -0.5,
    )
    ax = plot(
        x2,
        y2,
        data_name="my_data_2",
        x_name="my_x",
        y_name="my_y",
        title="my_title",
        new_fig=False,
    )
    ax = plot(
        x2,
        y2 + 0.5,
        data_name="my_data_2",
        x_name="my_x",
        y_name="my_y",
        title="my_title",
        new_fig=False,
    )


def _test_for_multi_fig():
    n = 1_000_000
    x1 = np.linspace(0, 2 * np.pi, n)
    y1 = np.sin(x1)
    x2 = np.linspace(0, 2 * np.pi, n)
    y2 = np.cos(x2)

    ax = plot(
        x1,
        y1,
        data_name="my_data_1",
        x_name="my_x",
        y_name="my_y",
        title="my_title",
        mask=y1 > -0.5,
    )
    ax = plot(
        x2,
        y2,
        data_name="my_data_2",
        x_name="my_x",
        y_name="my_y",
        title="my_title",
        new_fig=False,
    )
    ax = plot(
        x1,
        y1,
        data_name="my_data",
        x_name="my_x",
        y_name="my_y",
        title="my_title",
        share_x=ax,
    )
    ax = plot(
        x1,
        y1,
        color=y1,
        data_name="my_data",
        color_name="my_color",
        x_name="my_x",
        y_name="my_y",
        title="my_title",
        share_x=ax,
        share_y=ax,
    )


def _test_for_log_x():
    n = 1_000_000
    x1 = np.linspace(0, 2 * np.pi, n)
    y1 = np.sin(x1)

    ax = plot(
        np.exp(x1),
        np.exp(y1),
        data_name="my_data",
        x_name="my_x",
        y_name="my_y",
        title="my_title",
        log_x=True,
        log_y=True,
    )
    ax = plot(
        np.exp(x1),
        np.exp(y1),
        color=y1,
        data_name="my_data",
        color_name="my_color",
        x_name="my_x",
        y_name="my_y",
        title="my_title",
        log_x=True,
        share_x=ax,
    )


def _test_for_time_range():
    t = np.linspace(0, 2 * np.pi, 1_000_000)
    x = np.sin(t)
    y = np.cos(t)
    ax = plot(
        x,
        y,
        color=t,
        x_name="X Axis",
        y_name="Y Axis",
        color_name="Time",
        equal_scale=True,
        time_range=[1, 2],
    )


def _test_for_share_x_and_equal_scale():
    t = np.linspace(0, 2 * np.pi, 10_000)
    x = np.sin(t) + 10
    y = np.cos(t) + 10
    ax = None
    ax = plot(
        x,
        y,
        x_name="X Axis",
        y_name="Y Axis",
        data_name="data 1",
        equal_scale=True,
    )
    ax = plot(
        x,
        y + 1,
        x_name="X Axis",
        y_name="Y Axis",
        data_name="data 2",
        equal_scale=True,
        new_fig=False,
    )
    ax = plot(
        x,
        y,
        x_name="X Axis",
        y_name="Y Axis",
        data_name="data 1",
        equal_scale=True,
        share_x=ax,
        share_y=ax,
    )
    ax = plot(
        10**x,
        10**y,
        x_name="X Axis",
        y_name="Y Axis",
        data_name="data 2",
        equal_scale=True,
        log_x=True,
        log_y=True,
    )


def _test_for_data_info():
    n = 1_000_000
    x1 = np.linspace(0, 2 * np.pi, n)
    y1 = np.sin(x1)
    x2 = np.linspace(0, 2 * np.pi, n)
    y2 = np.cos(x2)
    data_info = {
        "info 1": y2,
        "info 2": y2 > 0,
        "info 3": ["hello"] * n,
        "info 4": [100] * n,
    }

    ax = plot(
        x1,
        y1,
        data_name="my_data_1",
        x_name="my_x",
        y_name="my_y",
        title="my_title",
        mask=y1 > -0.5,
    )
    ax = plot(
        x2,
        y2,
        data_name="my_data_2",
        x_name="my_x",
        y_name="my_y",
        title="my_title",
        new_fig=False,
        data_info=data_info,
    )
    ax = plot(
        x1,
        y1,
        color=y1,
        data_name="my_data",
        color_name="my_color",
        x_name="my_x",
        y_name="my_y",
        title="my_title",
        share_x=ax,
        share_y=ax,
        data_info=data_info,
    )


def _test_clip_view():
    point = np.array(
        [
            [0, 0],
            [1, 1],
            [1, 100],
            [1.5, 100],
            [2, 100],
            [2, 2],
            [3, 3],
        ]
    )
    x = point[:, 0]
    y = point[:, 1]
    plot(x, y)


def _test_mask():
    n = 100_000
    x = np.zeros(n)
    y = np.zeros(n)
    m = 1
    y[:m] = 1
    y[10_000 : 10_000 + m] = 1
    y[20_000:50_000] = 1
    y[-m:] = 1
    plot(x, data_name="x")
    plot(y, data_name="y", mask=y > 0, new_fig=False)


def _test_fig_num():
    try:
        plot([1, 2, 3], data_name="data 1", fig_num=101, new_fig=True)
    except Exception as e:
        print(f"data 1: {str(e)}")
    try:
        plot([1, 2, 3], data_name="data 2", fig_num=101, new_fig=False)
    except Exception as e:
        print(f"data 2: {str(e)}")
    try:
        plot([1, 2, 3], data_name="data 3", fig_num=102, new_fig=False)
    except Exception as e:
        print(f"data 3: {str(e)}")
    try:
        plot([1, 2, 3], data_name="data 4", fig_num=102, new_fig=True)
    except Exception as e:
        print(f"data 4: {str(e)}")
    try:
        plot([1, 2, 3], data_name="data 5", fig_num=102, new_fig=True)
    except Exception as e:
        print(f"data 5: {str(e)}")


def _test_for_compress():
    x = np.concatenate([np.zeros(100_000), np.arange(100_000)])
    y = x
    plot(x, y)


if __name__ == "__main__":
    _disable_debug()
    _enable_debug()
    close_figure()

    _test_for_single_fig()
    _test_for_multi_fig()
    _test_for_log_x()
    _test_for_time_range()
    _test_for_share_x_and_equal_scale()
    _test_for_data_info()
    _test_clip_view()
    _test_mask()
    _test_fig_num()
    _test_for_compress()

    show_figure()
