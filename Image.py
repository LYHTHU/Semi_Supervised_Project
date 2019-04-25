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

    # def show(self):
    #     cv2.imshow("image", self.img)
    #     cv2.waitKey(0)


if __name__ == '__main__':
    img = Image(True)
    img.load("./a.JPEG", 0)
    img1 = cv2.cvtColor(img.img, cv2.COLOR_BGR2GRAY).flatten()
    print(img1)
    feature = []
