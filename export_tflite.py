"""
scrfd_500m_640x640: Estimated count of arithmetic ops: 1.484 G  ops, equivalently 0.742 G  MACs
scrfd_500m_480x640: Estimated count of arithmetic ops: 1.113 G  ops, equivalently 0.556 G  MACs
scrfd_500m_256_320: Estimated count of arithmetic ops: 296.758 M  ops, equivalently 148.379 M  MACs
scrfd_500m_128_160: Estimated count of arithmetic ops: 74.189 M  ops, equivalently 37.095 M  MACs
"""
import argparse
import os
import cv2
import tensorflow as tf
import numpy as np
import glob

def representative_dataset(netin_height, netin_width):
    source_dir = os.path.join("data", "WIDER_val", "images")
    path_list = glob.glob(os.path.join(source_dir, "*", "*.jpg"))[:20]
    num_images = len(path_list)
    print("calibrate from {} images".format(num_images))
    for image_filepath in path_list:
        image = cv2.imread(image_filepath)
        netin = scrfd_preprocess(image, imshape=(netin_height, netin_width), expand=True, from_cv2=True)
        yield [netin]

#def representative_dataset(netin_height, netin_width):
#    for i in range(10):
#        val = np.random.randn(1, netin_height, netin_width, 3).astype(np.float32)
#        yield [val]

def scrfd_preprocess(image, imshape=(320, 320), expand=True, from_cv2=True):
    height = imshape[0]
    width = imshape[1]
    if height == -1:
        print("get -1 height/width, set 320x320 automatically")
        height = width = 320
    rs_image = cv2.resize(image, (width, height))
    if from_cv2:
        image_rgb = cv2.cvtColor(rs_image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image.copy()
    # normalization
    netin = (image_rgb - 127.5) / 128
    netin = np.expand_dims(netin, 0).astype(np.float32)
    return netin

def get_shape_suffix(saved_model_dir):
    """models/saved_model_480x640
    """
    basename = os.path.basename(saved_model_dir)
    except_shape = basename.split("_")[-1]
    try:
        height = int(except_shape.split("x")[0])
        width = int(except_shape.split("x")[1])
        print("parser shape from saved_model_dir succ: heightxwidth: {}x{}".format(height, width))
        return f"{height}x{width}_", height, width
    except:
        print("parser shape frome saved_model_dir fail: " + saved_model_dir)
        return "", -1, -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        help="input saved_model_dir",
                        default="models/saved_model_480x640")
    parser.add_argument("--dest_dir", type=str,
                        help='tflite model save_dir',
                        default="models")
    parser.add_argument("--int", action='store_true')
    args = parser.parse_args()

    model_dir = args.model_dir
    print("read model from: " + model_dir)
    shape_suffix, netin_height, netin_width = get_shape_suffix(model_dir)
    INT = args.int
    print("convert to int tflite: {}".format(INT))

    if INT:
        tflite_filename = f"scrfd500m_{shape_suffix}int8.tflite"
        tflite_filepath = os.path.join(args.dest_dir, tflite_filename)
    else:
        tflite_filename = f"scrfd500m_{shape_suffix}float32.tflite"
        tflite_filepath = os.path.join(args.dest_dir, tflite_filename)
    print("write result to: " + tflite_filepath)
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
    if INT:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(netin_height, netin_width)

    tflite_model = converter.convert()
    # Save the model.
    with open(tflite_filepath, 'wb') as f:
        f.write(tflite_model)