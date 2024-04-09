"""prepare modified onnx model for onnx2tf package include:
- update opset ver
- set input shape

# usage:
    $ python export_modified_onnx.py --hw=128,160
"""
import argparse
import os
import onnx

from utils_export import get_opset_ver, update_opset_ver, set_scrfd500m_in_out_dims

MODEL_PATH = os.path.join("models", "det_500m.onnx")



def parse_input_shape(input_shape_str):
    try:
        input_shape = tuple(map(int, input_shape_str.split(',')))
        if len(input_shape) != 2:
            raise ValueError("Input shape must have exactly two dimensions (height, width).")
        return input_shape
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))

def main():
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--hw', type=parse_input_shape, default=(320, 320),
                        help='Input shape of the image as "height,width". Default is 320, 320')

    args = parser.parse_args()
    print("read model from: " + MODEL_PATH)
    onnx_model = onnx.load(MODEL_PATH)
    opset_ver = get_opset_ver(onnx_model=onnx_model)
    print("source opset_ver: {}".format(opset_ver))
    updated_model = update_opset_ver(onnx_model=onnx_model)
    opset_ver = get_opset_ver(onnx_model=updated_model)

    model_name = os.path.basename(MODEL_PATH).strip('.onnx')
    print("Input shape:", args.hw)
    result_name = model_name + "_{}x{}.onnx".format(args.hw[0], args.hw[1])
    result_path = os.path.join("models", result_name)
    set_scrfd500m_in_out_dims(args.hw, onnx_model=updated_model, result_path=result_path)

if __name__ == "__main__":
    main()
