import argparse
import torch
from torchvision.utils import save_image

from net import UNet
from utils import keep_image_size_open


def run(weights: str, image: str, save_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet().to(device)
    if weights:
        model.load_state_dict(torch.load(weights))
        print(f'using weights {weights}')

    image = keep_image_size_open(image)
    out_image = model(image).to(device)
    show_img = torch.stack([image, out_image], 0)
    save_image(show_img, f'{save_path}/1.png')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='params/best.pt')
    parser.add_argument('--image', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
