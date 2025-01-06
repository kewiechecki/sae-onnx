import onnx
import os
from sae import Sae

def main():
    sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.10")

    # Specify the output directory and file name
    output_dir = "./onnx_model"
    model_path = "./onnx_model/sae.onnx"

    sae.export(output_dir)

    try:
        onnx.checker.check_model(model_path)
        print("The ONNX model is valid!")
    except onnx.checker.ValidationError as e:
        print(f"The ONNX model is invalid: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
