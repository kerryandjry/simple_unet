import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data import *
from net import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = r'params/114.pt'
data_path = 'VOC2012'
save_path = 'train_image'

if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path), batch_size=8, shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight')
    else:
        print('no weight')

    opt = optim.Adam(net.parameters())
    loss = nn.BCELoss()

    epoch = 1
    best_loss = 0
    while True:
        less_one_loss = 0
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image).to(device)
            train_loss = loss(out_image, segment_image)

            if train_loss < 0.15:
                less_one_loss += 1

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]
            image = torch.stack([_image, _segment_image, _out_image], 0)

            if i % 5 == 0:
                print(f'{epoch}-{i}-train_loss ===>>{train_loss}')

        print(f'best_loss = {best_loss}, less_one_loss = {less_one_loss}')
        if less_one_loss > best_loss:
            best_loss = less_one_loss
            torch.save(net.state_dict(), f'params/best.pt')
            print(f'save weight success!')

        save_image(image, f'train_image/{epoch}.png')

        epoch += 1
