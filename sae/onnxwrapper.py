import torch
import onnx
from torch import nn
import os

class SaeONNXWrapper(nn.Module):
    def __init__(self, sae):
        super().__init__()
        self.sae = sae

    def forward(self, x):
        return self.sae.forward_onnx(x)  # Uses the newly defined forward_onnx method
    def export(self, output_dir):
        torch_input = torch.randn(1, self.sae.d_in).cpu()

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "sae.onnx")

        torch.onnx.export(
            self,
            torch_input,
            output_path,
            export_params=True,
            opset_version=15,  # Ensure this is compatible with your environment
            input_names=["input"],
            output_names=["sae_out"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "sae_out": {0: "batch_size"}
            },
            # Enable external data format to handle large models
            use_external_data_format=True,
            do_constant_folding=True,
            verbose=True
        )

        print(f"Model successfully exported to {output_path}")

