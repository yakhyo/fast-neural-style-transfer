import argparse

import torch
import torch.onnx

import utils
from model import TransformerNet


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = utils.content_transform

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model)
            style_model.load_state_dict(state_dict)

            style_model.to(device)
            style_model.eval()
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, content_image, args.export_onnx, opset_version=11).cpu()
            else:
                output = style_model(content_image).cpu()
    utils.save_image(args.output_image, output[0])


def onnx_export(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = utils.content_transform

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    style_model = TransformerNet()
    state_dict = torch.load(args.model)
    style_model.load_state_dict(state_dict)

    style_model.to(device)
    style_model.eval()

    assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
    output = torch.onnx._export(style_model, content_image, args.export_onnx, opset_version=11).cpu()


def stylize_onnx(content_image, args):
    """
    Read ONNX model and run it using onnxruntime
    """

    assert not args.export_onnx

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(args.model)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(content_image)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    return torch.from_numpy(img_out_y)


def parse_args():
    parser = argparse.ArgumentParser(description="Stylizing parser for fast-neural-style")
    parser.add_argument(
        "--content-image",
        type=str,
        required=False,
        default="images/content-images/amber.jpg",
        help="path to content image you want to stylize")
    parser.add_argument("--content-scale", type=float, default=None, help="factor for scaling down the content image")
    parser.add_argument("--output-image", type=str, required=True, help="path for saving the output image")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="saved model to be used for stylizing the image"
    )
    parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
    parser.add_argument('--mps', action='store_true', default=False, help='enable macOS GPU training')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    stylize(args)


if __name__ == '__main__':
    main()
