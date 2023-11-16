
import csv

from torch import nn, optim
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image
from eval import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = 'Data_FAZ'
save_path = 'train_image'

if __name__ == '__main__':
    num_classes = 3
    data_loader = DataLoader(MyDataset(data_path), batch_size=1, shuffle=True)
    net = UNet(num_classes).to(device)

    # 判断权重文件是否存在
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))  # 可以读取权重
        print('successful load weight！')
    else:
        print('not successful load weight')

    # 优化器和Loss function设置
    opt = optim.Adam(net.parameters(), lr=0.0001)
    loss_fun = nn.BCELoss()

    epoch = 1
    while epoch < 1000:
        for i, (image, segment_image) in enumerate(data_loader):

            # forward
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            # backward
            opt.zero_grad()  # 将梯度归零
            train_loss.backward()  # 反向传播计算每个参数的梯度值

            # update weights
            opt.step()

            # 性能评估
            dice_tmp = dice_coefficient(out_image.cpu().detach().numpy(), segment_image.cpu().detach().numpy())
            hd95_tmp = hd95(out_image, segment_image)
            # assd_tmp = assd(out_image, segment_image)

            # 保存训练时候的各种loss
            if i % 5 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
                with open('train_BCELoss.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([i + 1, train_loss.item()])
                with open('train_dice.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([i + 1, dice_tmp.item()])
                with open('train_hd95.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([i + 1, hd95_tmp])
                # with open('train_assd.csv', 'a', newline='') as file:
                #     writer = csv.writer(file)
                #     writer.writerow([i + 1, assd_tmp.item()])

            # 保存模型参数
            if i % 10 == 0:
                torch.save(net.state_dict(), weight_path)
                print('save successfully!')

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            img = torch.stack([_segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')
        epoch += 1