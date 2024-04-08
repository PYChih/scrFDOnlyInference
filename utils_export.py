"""this files contain severial utils functions for export onnx models to tflite.
"""
import os
import onnx

def get_opset_ver(model_path=None, onnx_model=None):
    """get opset_ver from onnx model_path or onnx_model
    """
    if model_path is None:
        assert onnx_model is not None, "get both model_path and onnx_model both None"
    else:
        onnx_model = onnx.load(model_path)
    opset_ver = onnx_model.opset_import[0].version if len(onnx_model.opset_import) > 0 else None
    return opset_ver

def update_opset_ver(model_path=None, onnx_model=None, target_ver=18, result_path=None):
    """
    """
    if model_path is None:
        assert onnx_model is not None, "get both model_path and onnx_model both None"
    else:
        onnx_model = onnx.load(model_path)
    from onnx import version_converter
    converted_model = version_converter.convert_version(onnx_model, target_ver)
    if result_path is not None:
        print("write converted model to: {}".format(result_path))
        onnx.save(converted_model, result_path)
    return converted_model

def set_scrfd500m_in_out_dims(input_shape, output_dict=None, model_path=None, onnx_model=None, result_path=None):
    """
    Args:
        input_shape (tuple): (height, width)
    """
    if model_path is None:
        assert onnx_model is not None, "get both model_path and onnx_model both None"
    else:
        onnx_model = onnx.load(model_path)
    # param specific on scrfd_500m
    input_name = "input.1"
    stride_l = 32
    if output_dict is None:
        output_dict = {
            "493":[-1, 1],
            "496":[-1, 4],
            "499":[-1, 10],

            "468":[-1, 1],
            "471":[-1, 4],
            "474":[-1, 10],

            "443":[-1, 1],
            "446":[-1, 4],
            "449":[-1, 10],
        }
    height, width = input_shape[:2]
    print("set input shape(height, width): {}:{}".format(height, width))
    input_dict = {
        input_name : [1, 3, height, width]
    }
    from onnx.tools import update_model_dims
    variable_length_model = update_model_dims.update_inputs_outputs_dims(onnx_model, input_dict, output_dict)

    if result_path is not None:
        print("write converted model to: {}".format(result_path))
        onnx.save(variable_length_model, result_path)
    return variable_length_model

