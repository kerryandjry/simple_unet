import os

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, item):
        segment_name = self.name[item]
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('png', 'jpg'))
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image), transform(segment_image)


if __name__ == '__main__':
    data = MyDataset(r'VOC2008')
    print(len(data))
    # print(data[0][0])
    # print(data[0][1])
    # plt.imshow(data[0][1].permute(1, 2, 0))
    # plt.show()
