import numpy as np

class Scene:
    def __init__(self, points, segments):
        self.points = np.asarray(points, dtype=np.float64)
        self.segments = np.asarray(segments, dtype=np.int64)
        self.M = np.eye(4)

    def apply(self, m):
        self.M = self.M @ m

    def transformed_points(self):
        return self.points @ self.M

    def segments_xy(self):
        p = self.transformed_points()
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = p[:, :2] / p[:, 3:4]
        return xy[self.segments]