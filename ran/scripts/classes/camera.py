import numpy as np

class SimplePinholeCamera:
    def __init__(self, f=None, cx=None, cy=None):
        self.f = f
        self.cx = cx
        self.cy = cy

    def K(self):
        """Return intrinsic calibration matrix."""
        return np.array([
            [self.f,    0, self.cx],
            [0,    self.f, self.cy],
            [0,        0,       1]
        ], dtype=np.float64)

    def __repr__(self):
        return f"<SimplePinholeCamera f={self.f:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}>"