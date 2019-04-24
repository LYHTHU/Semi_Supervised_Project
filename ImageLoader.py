from os import listdir
from Image import Image


class ImageLoader:
    def __init__(self, data_root_path="./ssl_data_96"):
        self.data_path = data_root_path
        self.train = []
        self.val = []
        self.unsup = []
        self.cls2int = {}
        self.int2cls = {}
        self.count_cls = 0

    def load_supervised(self, dir="train"):
        ret = []
        root_path = self.data_path + "/supervised"
        crt_path = root_path + "/" + dir
        lst = 0.0
        for i, cls in enumerate(listdir(crt_path)):
            if cls not in self.cls2int:
                self.cls2int[cls] = self.count_cls
                self.int2cls[self.count_cls] = cls
                self.count_cls += 1

            label = self.cls2int[cls]

            cls_path = crt_path + "/" + cls

            if i / 1000 - lst > 0.05:
                lst = i / 1000
                print("Loading: ", int(lst * 100), "%")

            for img_name in listdir(cls_path):
                img = Image(supervised=True)
                img.load(cls_path+"/"+img_name, label)
                ret.append(img)
        return ret

    @staticmethod
    def parse_supervised(data):
        images, labels = [], []
        for img in data:
            images.append(img.img)
            labels.append(img.label)
        return images, labels

    def load_unsupervised(self):
        ret = []
        root_path = self.data_path + "/unsupervised"
        lst = 0.0
        count_unsup = 0
        list_dir_unsup = listdir(root_path)
        n_folder = len(list_dir_unsup)
        for i, dir in enumerate(list_dir_unsup):
            if i / n_folder - lst > 0.01:
                lst = i / n_folder
                print("Loading: unsupervised, ", int(lst * 100), "%")

            crt_path = root_path + "/" + dir
            print(len(listdir(crt_path)))

            for img_name in listdir(crt_path):
                img = Image(supervised=False)
                img.load(crt_path+"/"+img_name)
                ret.append(img)
                count_unsup += 1

        print(count_unsup)


if __name__ == '__main__':
    loader = ImageLoader()
    loader.train = loader.load_supervised("train")
    images, labels = ImageLoader.parse_supervised(loader.train)
    # loader.val = loader.load_supervised("val")
    # loader.unsup = loader.load_unsupervised()
