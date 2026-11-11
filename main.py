from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, RadioButtons, TextBox

import transforms as t
from insert import get_data
from scene import Scene

MARGIN = 1.2
FIT_FILL = 0.99


@dataclass
class Control:
    label: str
    build: Callable[[float], np.ndarray]
    group: str


def deg(factory):
    return lambda v: factory(np.radians(v))


CONTROLS = [
    Control("X, градусы", deg(t.rotation_x), "Поворот"),
    Control("Y, градусы", deg(t.rotation_y), "Поворот"),
    Control("Z, градусы", deg(t.rotation_z), "Поворот"),
    Control("X", lambda v: t.transfer(v, 0, 0), "Перенос"),
    Control("Y", lambda v: t.transfer(0, v, 0), "Перенос"),
    Control("Z", lambda v: t.transfer(0, 0, v), "Перенос"),
    Control("коэффициент k", t.zoom, "Масштаб"),
    Control("XY", t.obl_shift_xy, "Косой перенос"),
    Control("XZ", t.obl_shift_xz, "Косой перенос"),
    Control("YX", t.obl_shift_yx, "Косой перенос"),
    Control("YZ", t.obl_shift_yz, "Косой перенос"),
    Control("ZX", t.obl_shift_zx, "Косой перенос"),
    Control("ZY", t.obl_shift_zy, "Косой перенос"),
    Control("1/f по X", t.opp_x, "ОПП"),
    Control("1/f по Y", t.opp_y, "ОПП"),
    Control("1/f по Z", t.opp_z, "ОПП"),
]


def window_radius(scene):
    xyz = scene.points[:, :3] / scene.points[:, 3:4]
    r = float(np.linalg.norm(xyz, axis=1).max()) * MARGIN
    return r or 1.0


class Viewer:
    ROW_H = 0.05

    def __init__(self, scene, controls=None):
        self.scene = scene
        self.controls = list(controls or CONTROLS)
        self.R = window_radius(scene)

        self.groups = {}
        for i, c in enumerate(self.controls):
            self.groups.setdefault(c.group, []).append(i)
        self.rows = max(max(map(len, self.groups.values())), len(self.groups))

        self.fig = plt.figure(figsize=(10, 8))
        bottom = 0.08 + self.ROW_H * self.rows
        self.ax = self.fig.add_axes([0.10, bottom, 0.80, 0.95 - bottom])
        self.ax.set_xlim(-self.R, self.R)
        self.ax.set_ylim(-self.R, self.R)
        self.ax.set_aspect("equal", adjustable="box")
        self.lc = LineCollection(self._segments(), colors="tab:blue", linewidths=1.5)
        self.ax.add_collection(self.lc)

        self._build_widgets()
        self.select_group(next(iter(self.groups)))


    def _build_widgets(self):
        panel_h = self.ROW_H * self.rows - 0.01
        self.menu = RadioButtons(self.fig.add_axes([0.02, 0.05, 0.18, panel_h]),
                                 list(self.groups))
        self.menu.on_clicked(self.select_group)



        self.boxes = [None] * len(self.controls)
        self.labels = [None] * len(self.controls)
        for indices in self.groups.values():
            for row, i in enumerate(indices):
                y = 0.05 + self.ROW_H * (self.rows - 1 - row)
                self.labels[i] = self.fig.text(0.24, y + 0.015, self.controls[i].label,
                                               va="center", ha="left")
                box = TextBox(self.fig.add_axes([0.36, y, 0.12, 0.03]), "")
                box.on_submit(lambda text, i=i: self.on_text(i, text))
                self.boxes[i] = box

        self.buttons = [
            self._button("Вписать", [0.78, 0.10, 0.10, 0.04], lambda _: self.fit()),
        ]

    def _button(self, label, rect, callback):
        b = Button(self.fig.add_axes(rect), label)
        b.on_clicked(callback)
        return b

    def select_group(self, name):
        for group, indices in self.groups.items():
            visible = group == name
            for i in indices:
                self.boxes[i].set_active(visible)
                self.boxes[i].ax.set_visible(visible)
                self.labels[i].set_visible(visible)
        self.fig.canvas.draw_idle()


    def on_text(self, i, text):
        try:
            value = float(text.replace(",", ".").strip())
        except ValueError:
            value = np.nan
        self.boxes[i].set_val("")
        if np.isfinite(value):
            self.scene.apply(self.controls[i].build(value))
            self.update()

    def fit(self):
        xy = self.scene.raw_segments_xy().reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy):
            self.scene.fit_M = t.fit_matrix(xy.min(axis=0), xy.max(axis=0), self.R, FIT_FILL)
            self.update()

    def reset(self):
        self.scene.reset()
        self.update()


    def _segments(self):
        xy = self.scene.segments_xy()
        return np.where(np.isfinite(xy), xy, np.nan)

    def update(self):
        self.lc.set_segments(self._segments())
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    points, segments = get_data("data.txt")
    Viewer(Scene(points, segments)).show()


if __name__ == "__main__":
    main()