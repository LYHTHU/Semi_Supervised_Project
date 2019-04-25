from ImageLoader import ImageLoader
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from sklearn import svm
from sklearn.metrics import classification_report


if __name__ == '__main__':
    pass
    # Using torch loader
    # data = torchvision.datasets.ImageFolder(root="./ssl_data_96")
    # loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True, num_workers=8)

    cv_loader = ImageLoader()
    images, labels = ImageLoader.parse_supervised(cv_loader.load_supervised("train"))
    val, l_vals = ImageLoader.parse_supervised(cv_loader.load_supervised("val"))
    feature = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
        feature.append(img)

    val_feature = []
    for img in val:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
        val_feature.append(img)

    clf = svm.SVC(gamma='scale')
    clf.fit(feature, labels)
    pred = clf.predict(val_feature)
    print(classification_report(l_vals, pred))



