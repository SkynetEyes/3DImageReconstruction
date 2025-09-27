import numpy as np

class SimplePinholeCamera:
    def __init__(self, fx=None, fy=None, cx=None, cy=None):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def K(self):
        """Return intrinsic calibration matrix."""
        return np.array([
            [self.fx,    0,       self.cx],
            [0,          self.fy, self.cy],
            [0,          0,       1      ]
        ], dtype=np.float64)

    def __repr__(self):
        return f"<SimplePinholeCamera fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}>"