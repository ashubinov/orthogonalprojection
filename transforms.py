import numpy as np

def rotation_x(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1, 0, 0, 0],
                     [0, c, s, 0],
                     [0, -s, c, 0],
                     [0, 0, 0, 1]], dtype=float)


def rotation_y(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, 0, -s, 0],
                     [0, 1, 0, 0],
                     [s, 0, c, 0],
                     [0, 0, 0, 1]], dtype=float)


def rotation_z(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, s, 0, 0],
                     [-s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)


def transfer(x, y, z):
    M = np.eye(4)
    M[3, :3] = (x, y, z)          # было M[:3, 3] — это конвенция для столбцов
    return M


def zoom(k):
    M = np.eye(4)
    M[0, 0] = M[1, 1] = M[2, 2] = k
    return M


def _shear(i, j, k):
    M = np.eye(4)
    M[i, j] = k
    return M


def obl_shift_xy(k): return _shear(1, 0, k)
def obl_shift_xz(k): return _shear(2, 0, k)
def obl_shift_yx(k): return _shear(0, 1, k)
def obl_shift_yz(k): return _shear(2, 1, k)
def obl_shift_zx(k): return _shear(0, 2, k)
def obl_shift_zy(k): return _shear(1, 2, k)


def _persp(axis, p):
    M = np.eye(4)
    M[axis, 3] = p
    return M


def opp_x(p): return _persp(0, p)
def opp_y(p): return _persp(1, p)
def opp_z(p): return _persp(2, p)


def fit_matrix(lo, hi, R, fill=0.9):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    center = (lo + hi) / 2
    extent = float(np.max(hi - lo))
    lam = 2 * R * fill / extent if extent > 1e-12 else 1.0
    return transfer(-center[0], -center[1], 0) @ zoom(lam)   # сначала перенос, потом масштаб
