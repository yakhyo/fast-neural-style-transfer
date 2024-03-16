import argparse
import os
import sys
import numpy as np

import torch
import torch.onnx
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets

import utils
from model import TransformerNet, VGG16


def train(device, args):
    np.random.seed(0)
    torch.manual_seed(0)

    train_dataset = datasets.ImageFolder(args.dataset, transform=utils.train_transform(args.image_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8)

    # Define networks and move to device
    transformer = TransformerNet().to(device)
    vgg = VGG16(requires_grad=False).to(device)

    # Define optimizer and loss
    optimizer = optim.Adam(transformer.parameters(), args.lr)
    l2_loss = torch.nn.MSELoss()

    style_transform = utils.style_transform()
    style = utils.load_image(args.style_image, size=args.style_size)

    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # Extract style features
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for epoch in range(args.epochs):
        transformer.train()
        metrics = {"content_loss": [], "style_loss": [], "total_loss": [], "count": 0}

        for i, (images, _) in enumerate(train_loader):
            metrics["count"] += images.size(0)
            optimizer.zero_grad()

            images = images.to(device)
            images_transformed = transformer(images)

            images = utils.normalize_batch(images)
            images_transformed = utils.normalize_batch(images_transformed)

            # Extract features
            features = vgg(images)
            features_transformed = vgg(images_transformed)

            # Compute content los as MSE between features
            content_loss = args.content_weight * l2_loss(features_transformed.relu2_2, features.relu2_2)

            # Compute style loss as MSE between gram matrices
            style_loss = 0.
            for ft_y, gm_s in zip(features_transformed, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += l2_loss(gm_y, gm_s[:images.size(0), :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            metrics["content_loss"].append(content_loss.item())
            metrics["style_loss"].append(style_loss.item())
            metrics["total_loss"].append(total_loss.item())

            if (i + 1) % args.log_interval == 0:
                msg = "[Epoch {}/{}]|[{}/{}] | [Content Loss: {:.2f} | Style Loss: {:.2f} | Total Loss: {:.2f}]".format(
                    epoch + 1,
                    args.epochs,
                    metrics["count"],
                    len(train_dataset),
                    np.mean(metrics["content_loss"]),
                    np.mean(metrics["style_loss"]),
                    np.mean(metrics["total_loss"])
                )
                print(msg)

    # save model
    transformer.eval().cpu()
    filename = f"{os.path.splitext(os.path.basename(args.style_image))[0]}.pth"
    path = os.path.join(args.save_model, filename)
    torch.save(transformer.state_dict(), path)

    print(f"\nDone, trained model saved at {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Training parser for fast-neural-style")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to training dataset"
    )
    parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg", help="path to style-image")
    parser.add_argument("--epochs", type=int, default=1, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size for training")
    parser.add_argument("--image-size", type=int, default=256, help="size of training images")
    parser.add_argument(
        "--style-size",
        type=int,
        default=None,
        help="size of style-image, default is the original size of style image"
    )

    parser.add_argument("--save-model", type=str, required=True, help="folder to save model weights")
    parser.add_argument("--content-weight", type=float, default=1e5, help="weight for content-loss")
    parser.add_argument("--style-weight", type=float, default=1e10, help="weight for style-loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="number of images after which the training loss is logged"
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        os.makedirs(args.save_model, exist_ok=True)
    except OSError as e:
        print(e)
        sys.exit(1)

    train(device, args)


if __name__ == "__main__":
    main()
