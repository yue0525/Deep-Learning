import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T
import torchvision.transforms as transforms
import os


def get_path(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        # label = pd.read_csv('train_label.csv')
        img_name = np.squeeze(img.values)
        # label = np.squeeze(label.values)
        for index in range(len(img_name)):
            path = os.path.join("./data//new_train", img_name[index] + '.jpeg')
            image(path)
    else:
        img = pd.read_csv('test_img.csv')
        img_name = np.squeeze(img.values)
        for index in range(len(img_name)):
            path = os.path.join("./data//new_test", img_name[index] + '.jpeg')
            image(path)
    # path = os.path.join(self.root, self.img_name[index] + '.jpeg')

    # img = Image.open(path)
    return path


def image(path):
    print(path)
    img = read_image(path)

    # img_tensor = transforms.ToTensor()(img)
    C, H, W = img.size()
    # 中心裁剪为 224x224 大小的图像
    if H == W:
        return
    if H < W:
        center_cropped = transforms.CenterCrop((H, H))(img)
    if H > W:
        center_cropped = transforms.CenterCrop((W, W))(img)
    # 将裁剪后的 Tensor 转换为 PIL Image，并保存
    center_cropped_img = transforms.ToPILImage()(center_cropped)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 缩放到 256x256 大小
        transforms.ToTensor(),  # 转换为 Tensor
    ])

    img_transformed = transform(center_cropped_img)

    # 输出变换后的 Tensor 大小
    # print(img_transformed.size())  # 输出 torch.Size([3, 256, 256])
    center_cropped_img = transforms.ToPILImage()(img_transformed)
    # center_cropped_img.show()
    center_cropped_img.save(path)


if __name__ == '__main__':
    # read a JPEG image
    get_path("train")
    get_path("test")
    # image("./data//new_test//3798_left.jpeg")
