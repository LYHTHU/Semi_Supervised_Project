import cv2


class Image:
    def __init__(self, supervised=False):
        self.img = None
        self.label = None
        self.supervised = supervised

    def load(self, path, label=None):
        if self.supervised:
            self.label = label
        self.img = cv2.imread(path)

    def show(self):
        cv2.imshow(self.img)
