import torch
import onnx
from sae import Sae

d_in = 4096
sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.10")
sae.cuda()

torch_input = torch.randn(1, d_in).cuda()
output = sae.forward(torch_input)
print(output)

#torch.onnx.export(sae.forward_for_onnx,torch_input,"sae.onnx")

torch.onnx.export(
    sae,
    (torch_input,),
    "sae.onnx",
    export_params=True,
    input_names=["input"],
    output_names=["sae_out"],
    forward_method="forward_onnx",  # <--- specify your custom forward
)

