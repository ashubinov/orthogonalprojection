import numpy as np

class Scene:
    def __init__(self, points, segments):
        self.points = np.asarray(points, dtype=np.float64)
        self.segments = np.asarray(segments, dtype=np.int64)
        self.M = np.eye(4)
        self.fit_M = np.eye(4)  # НОВОЕ: отдельная матрица для вписывания

    def apply(self, m):
        self.M = self.M @ m

    def reset(self):
        self.M = np.eye(4)
        self.fit_M = np.eye(4)  # Сбрасываем и вписывание

    def transformed_points(self):
        return self.points @ self.M @ self.fit_M

    def raw_segments_xy(self):
        p = self.points @ self.M
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = p[:, :2] / p[:, 3:4]
        return xy[self.segments]

    def segments_xy(self):
        # Итоговая проекция с учетом вписывания
        p = self.transformed_points()
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = p[:, :2] / p[:, 3:4]
        return xy[self.segments]