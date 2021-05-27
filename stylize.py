import time

import yaml
import argparse

import torch
from nets.nn import TransformerNet

from utils import util


def stylize_image(config, device):
    config = config['VAL']

    # Load image
    input_image = util.load_image(config['content_image'], scale=config['content_scale'])

    # Transform input image
    input_image = util.transform(input_image)
    input_image = input_image.unsqueeze(0).to(device)

    model_name = config['model'].split("/")[-1][:-4]
    image_name = config['content_image'].split("/")[-1][:-4]

    with torch.no_grad():
        net = TransformerNet().to(device)
        net.load_state_dict(torch.load(config['model']))
        output = net(input_image).cpu()
    output_image = f"{config['output_path']}/{image_name}_{model_name}_{int(time.time())}.jpg"
    util.save_image(output_image, output[0])


if __name__ == "__main__":
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configs
    with open(r'utils/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    stylize_image(config=config, device=device)
