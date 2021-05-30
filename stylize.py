import os
import time

import yaml
import argparse

import torch
from nets.nn import TransformerNet

from utils import util


def stylize_image(config, device, args):
    config = config['STYLIZE']

    # Load image
    input_image = util.load_image(args.img, scale=config['content_scale'])

    # Transform input image
    input_image = util.transform(input_image)
    input_image = input_image.unsqueeze(0).to(device)

    image_name = args.img.split("/")[-1][:-4]
    net = TransformerNet().to(device)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    for i in os.listdir(config['models_path']):
        path = f"{config['models_path']}/{i}"
        model_name = path.split("/")[-1][:-4]
        net.load_state_dict(torch.load(path))
        with torch.no_grad():
            output = net(input_image).cpu()
        output_image = f"{config['output_path']}/{image_name}_{model_name}.jpg"
        # output_image = f"{config['output_path']}/{image_name}_{model_name}_{int(time.time())}.jpg"
        util.save_image(output_image, output[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, type=str, help='path to an image to stylize')
    args = parser.parse_args()

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configs
    with open(r'utils/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    stylize_image(config=config, device=device, args=args)
