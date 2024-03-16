import argparse

import torch
import torch.onnx

import logging
import utils
from model import TransformerNet

logging.basicConfig(format='[%(levelname)s]:%(message)s', level=logging.INFO)


def content_image_preprocess(args):
    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_image = utils.content_transform(content_image)
    content_image_tensor = content_image.unsqueeze(0)

    return content_image_tensor


def stylize(model, device, args):
    content_image_tensor = content_image_preprocess(args).to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(content_image_tensor).cpu()

    utils.save_image(args.output_image, output[0])


def onnx_export(model, device, args):
    content_image_tensor = content_image_preprocess(args).to(device)

    model.to(device)
    model.eval()

    assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
    torch.onnx.export(
        model,
        content_image_tensor,
        args.export_onnx,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=11
    )


def stylize_onnx(device, args):
    """Read ONNX model and run it using onnxruntime
    Args:
        device: device to run the model on
        args: parser arguments
    """

    assert not args.export_onnx

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(
        args.model,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    content_image_tensor = content_image_preprocess(args).to(device)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(content_image_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    output = torch.from_numpy(img_out_y)
    utils.save_image(args.output_image, output[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Stylizing parser for fast-neural-style")
    parser.add_argument("--content-image", type=str, required=True, help="path to content image you want to stylize")
    parser.add_argument("--content-scale", type=float, default=None, help="factor for scaling down the content image")
    parser.add_argument("--output-image", type=str, required=True, help="path for saving the output image")
    parser.add_argument("--model", type=str, required=True, help="saved model to be used for stylizing the image")
    parser.add_argument("--export-onnx", type=str, help="export ONNX model to a given file")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model.endswith(".pth"):
        model = TransformerNet()
        model.load_state_dict(torch.load(args.model))
        if args.export_onnx:
            logging.info("Export To ONNX")
            onnx_export(model, device, args)
        else:
            logging.info(f"PyTorch Inference | Device: {device}")
            stylize(model, device, args)

    if args.model.endswith(".onnx"):
        logging.info(f"ONNX Inference")
        stylize_onnx(device, args)


if __name__ == '__main__':
    main()
