"""this file implement the blink module
official implement preprocessing using the eyes left/right corner
from: https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/gaze_estimation_demo/cpp/src/eye_state_estimator.cpp
input/output define here: https://docs.openvino.ai/2022.3/omz_models_model_open_closed_eye_0001.html
here we try to modify the input preprocessing from 5pt lms
"""
import os
import cv2
import numpy as np
import tensorflow as tf


class blink(object):
    def __init__(self):
        self.model_path = os.path.join("models", "blink", "blink_32x32x3_float32.tflite")
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.face_elements = ['left_eye', 'right_eye', 'face']
        # get tflite model input shape
        netin_shape = self.input_details[0]['shape']  # ndarray with shape (1, netin_height, netin_width, netin_c)
        print("-----[blink]netin_shape: {}".format(netin_shape))
        self.netin_height = netin_shape[1]
        self.netin_width = netin_shape[2]
        self.netin_c = netin_shape[3]
        self.netin_shape = (self.netin_height, self.netin_width, netin_shape[3])

    def outer_preprocess(self, cam_in, lms, debug=False):
        """Given Camera image and corresponding landmark pts crop and resize to netin

        Args:
            cam_in (ndarray): usually in the shape (720, 1280, 3) or (480, 640, 3)
            lms (ndarray): with shape (5, 2)
            norm (bool): normalize the image or not
        
        Returns:
            netin_images (list): of ndarray with len 3, in the order [left_eye, right_eye, croped_face]
        
        Note:
            implement based on 3DGazeNet-Demo
        """
        input_args = (cam_in, [self.netin_width, self.netin_height], 0, False)  # 0: rotation, False: flip
        diag1 = np.linalg.norm(lms[0] - lms[4])
        diag2 = np.linalg.norm(lms[1] - lms[3])
        diag = np.max([diag1, diag2])
        if debug:
            print("[gazenet]outer_preprocessing")
            print("    diag1: {:.2f}, diag2: {:.2f}, diag: {:2f}".format(diag1, diag2, diag))
        face_crop_len = int(1.5 * diag)
        eyes_crop_len = int(2 * diag / 5)
        self.eyes_crop_len = eyes_crop_len
        centers = [lms[1], lms[0], lms[2]]  # left_eyes, right_eyes, nose
        self.centers = centers
        crop_info = {
            'left_eye' : {'center': centers[0], 'crop_len': [eyes_crop_len, eyes_crop_len]},
            'right_eye': {'center': centers[1], 'crop_len': [eyes_crop_len, eyes_crop_len]},
            'face'     : {'center': centers[2], 'crop_len': [face_crop_len, face_crop_len]}
        }
        netin_images = []
        trans_list = []
        for eye_str in self.face_elements:
            center = crop_info[eye_str]['center']
            crop_len = crop_info[eye_str]['crop_len']
            # resize to model preferences
            trans, img_patch_cv = get_input_and_transform(center, crop_len, *input_args)
            netin_images += [img_patch_cv]
            trans_list += [trans]
        # invert transforms image -> patch
        trans_list_inv = []
        for i in range(2):
            try:
                trans_list[i] = np.concatenate((trans_list[i], np.array([[0, 0, 1]])))
                trans_list_inv += [np.linalg.inv(trans_list[i])[:2]]
            except:
                print(f'Error inverting bbox crop transform')
                return None
        self.trans_list_inv = trans_list_inv
        return netin_images
    
    def _preprocess(self, netin_images, BGR2RGB=False):
        """Generate network input(netin) from croped_image
        here we inference via batch-size == 1
        - can modify batch-size in onnx format
        """
        netin_lst = []
        for image_idx, image in enumerate(netin_images):
            if image_idx == 2:
                break
            im_height, im_width, _ = image.shape
            assert im_height == self.netin_height, f"except netin height: {self.netin_height}, but get: {im_height}"
            assert im_width == self.netin_width, f"except netin width: {self.netin_width}, but get: {im_width}"
#            netin = image/255.
            netin = (image - 127.5) / 128
            netin = np.expand_dims(netin, 0).astype(np.float32)
            netin_lst.append(netin)
        return netin_lst

    def _invoke(self, netin_lst):
        """Pure model inference

        Args:

        Returns:
       
        Note:
            - Sometimes tflite model has different output node order should fix
            - always suppose batch size equals to 1.
        """
        rvs = []
        for netin in netin_lst:
            self.interpreter.set_tensor(self.input_details[0]['index'], netin)
            self.interpreter.invoke()
            eye_state = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # (1, 1, 2)
            rvs.append(eye_state)
        return rvs
    
    def _postprocess(self, netout):
        """
        """
        rvs = []
        for eye_state in netout:
            eye_state = eye_state[0, 0, :]
            if eye_state[1] > eye_state[0]:
                rvs.append('open')
            else:
                rvs.append('close')
        return rvs

def get_input_and_transform(center, width_height, cv_img_numpy, crop_size, rotation, do_flip):
    img_height, img_width, img_channels = cv_img_numpy.shape
    if do_flip:
        cv_img_numpy = cv_img_numpy[:, ::-1, :]
        center[0] = img_width - center[0] - 1
    trans = gen_trans_from_patch_cv_tuples(center, width_height, crop_size, rotation, inv=False)
    input = cv2.warpAffine(cv_img_numpy, trans, tuple(crop_size), flags=cv2.INTER_LINEAR)
    return trans, input

def gen_trans_from_patch_cv_tuples(center_xy, src_wh, dst_wh, rot, inv=False):
    return gen_trans_from_patch_cv(center_xy[0], center_xy[1],
                                   src_wh[0], src_wh[1],
                                   dst_wh[0], dst_wh[1],
                                   rot, inv)

def gen_trans_from_patch_cv(c_x, c_y, src_w, src_h, dst_w, dst_h, rot,
                            inv=False):
    rot_rad = np.pi * rot / 180
    #
    src_center = np.array([c_x, c_y], dtype=np.float32)
    src_downdir = _rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = _rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)
    #
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def _rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)