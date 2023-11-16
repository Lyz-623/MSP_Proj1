import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):

    # 构造器
    def __init__(self, path):
        self.path = path  # data address
        self.name = os.listdir(os.path.join(path, r'Domain1\train\mask'))  # 返回所有文件名

    # 获取数据长度
    def __len__(self):
        return len(self.name)

    # 通过key来取对应值
    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png
        segment_path = os.path.join(self.path, r'Domain1\train\mask', segment_name)
        image_path = os.path.join(self.path, r'Domain1\train\imgs', segment_name)
        segment_image = Image.open(segment_path)
        image = Image.open(image_path)
        return transform(image), transform(segment_image)  # 将图片转换为tensor


if __name__ == '__main__':
    data = MyDataset('Data_FAZ')
    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data.name)
