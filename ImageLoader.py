import os
import Image


class ImageLoader:
    def __init__(self, data_root_path="./ssl_data_96"):
        self.data_path = data_root_path

    def load_supervised(self):
        root_path = self.data_path+"/supervised"

    def load_unsupervised(self):
        root_path = self.data_path+"/unsupervised"