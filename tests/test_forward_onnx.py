import torch
from sae import Sae

def test_forward_onnx():
    d_in = 4096
    sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.10")
    sae.cpu()  # Move to CPU for testing
    torch_input = torch.randn(1, d_in).cpu()
    output = sae.forward_onnx(torch_input)
    print(output)

if __name__ == "__main__":
    test_forward_onnx()
