import numpy as np


def get_data(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = [ln.strip() for ln in file if ln.strip()]

    pos = 0
    n = int(lines[pos].split()[0])
    pos += 1
    points = np.array(
        [[float(x) for x in lines[pos + i].split()] for i in range(n)],
        dtype=np.float64,)
    pos += n

    m = int(lines[pos].split()[0])
    pos += 1
    segments = np.array(
        [[int(x) for x in lines[pos + i].split()[:2]] for i in range(m)],
        dtype=np.int64,)

    if segments.size and segments.min() >= 1 and segments.max() == n:
        segments -= 1

    return points, segments

