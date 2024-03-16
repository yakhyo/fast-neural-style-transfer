from model import TransformerNet
from torchvision.transforms import v2
import torch
import onnxruntime


class InferenceProcess:
    def __init__(self, model_path: str) -> None:
        self.ort_session = onnxruntime.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.model = TransformerNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _preprocess(image):
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((1080, 1080)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 255.0, 1 / 255.0, 1 / 255.0])
        ])

        # Apply transformations
        image_tensor = transform(image)
        # Add batch dimension
        image_batch = image_tensor.unsqueeze(0)

        return image_batch

    def __call__(self, image):
        preprocessed_input = self._preprocess(image)
        preprocessed_input = preprocessed_input.to(self.device)

        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(preprocessed_input)}
        ort_outs = self.ort_session.run(None, ort_inputs)
        img_out_y = ort_outs[0]

        output = torch.from_numpy(img_out_y)

        return output
