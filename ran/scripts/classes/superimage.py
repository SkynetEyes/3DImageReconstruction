import cv2
import math

class SuperImage:
    def __init__(self):
        self.imag_color = None
        self.imag_gray = None
        self.keypoints = None
        self.descripts = None

    def set_imag_color(self, imag_color):
        self.imag_color = imag_color
        return self

    def set_imag_gray(self, imag_gray):
        self.imag_gray = imag_gray
        return self

    def set_keypoints(self, keypoints):
        self.keypoints = keypoints
        return self

    def set_descriptors(self, descripts):
        self.descripts = descripts
        return self

    def imag_w_keypoints(self, circle_color=(0, 255, 0), circle_thickness=2, line_thickness=2):
        imag = self.imag_color.copy()

        for kpt in self.keypoints:
            x, y = int(kpt.pt[0]), int(kpt.pt[1])
            radius = int(kpt.size / 2)

            # Draw circle
            cv2.circle(imag, (x, y), radius, circle_color, thickness=circle_thickness)

            # Draw orientation line
            angle_rad = math.radians(kpt.angle)
            x2 = int(x + radius * math.cos(angle_rad))
            y2 = int(y + radius * math.sin(angle_rad))
            cv2.line(imag, (x, y), (x2, y2), circle_color, thickness=line_thickness)

        return imag