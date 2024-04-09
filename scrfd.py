"""this file implement the scrfd: fd+5ptlms module
camera_in->adjust_input->preprocess_image->_invoke->postprocess_output->adjust_output_scale->bndbox_and_landmark
camera_in->adjust_input->                  _inference                 ->adjust_output_scale->bndbox_and_landmark
camera_in->                                inference                                       ->bndbox_and_landmark
Note:
  - [ ] currently the tflite int file has large quantized error, but it can be solve by clip the conf output node(maybe), update soon.
  - [ ] it's seems cv2.dnn.NMS has some problem when all the input box has the same conf score
    - it return empty tuple instead of except ndarray, using self implement NMS or fix the quantized will solve the problem
    - currently, we just simply solve this problem by: if nms output empty tuple, chose the first pred box auto.
"""
import os
import cv2
import time
import numpy as np
import tensorflow as tf

from pprint import pprint

class ResizeAndPad:
    def __init__(self, netin_shape, center=True, fillv=114, debug=False):
        """Resize/crop/pad cam-in_image to netin_image, adjust_input for scrfd modules

        Args:
            netin_shape (tuple): (netin_height, netin_width, netin_c) the result image(netin_image) will be processed to this shape
            center (bool): keep image in the center by pad left(top) and right(bottom), or right-bottom pad
            fillv (int): padding value
            debug (bool): show the debug info or not
        """
        self.netin_height = netin_shape[0]
        self.netin_width = netin_shape[1]
        self.netin_c = netin_shape[2]
        self.netin_shape = (self.netin_height, self.netin_width, self.netin_c)

        self.debug = debug

        if self.debug:
            print("class ResizeAndPad-__init__: netin_shape: {}".format(netin_shape))

        self.center = center

        self.fillv = fillv
    
    def __call__(self, image):
        """
        Args:
            image (ndarray): source image(cam-in) to be processed, usually be cam-in with shape (720, 1280, 3) or (480, 640, 3)
        
        Returns:
            processed_image (ndarray): processed(resize and pad) image with netin shape
        
        Note:
            should keep ratio and pad info from unprocessed(netout)
        """
        self.source_height, self.source_width, self.source_c = image.shape
        r_height = self.netin_height / self.source_height
        r_width = self.netin_width / self.source_width
        self.r = min(r_height, r_width)
        # Compute padding
        unpad = int(round(self.source_width * self.r)), int(round(self.source_height * self.r))  # width x height for cv2.resize
        dh = self.netin_height - unpad[1]
        dw = self.netin_width - unpad[0]

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        img = cv2.resize(image, unpad, interpolation=cv2.INTER_LINEAR)  # [ ] nearest neighbor
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        if self.debug:
            print("class ResizeAndPad-__call__: r_height: {:.2f}, r_width: {:.2f}, r: {:.2f}".format(r_height, r_width, self.r))
            print("class ResizeAndPad-__call__: unpad image shape: (widthxheight): {}".format(unpad))
            print("class ResizeAndPad-__call__: processed image height pad: {}, width pad: {}".format(dh, dw))
            print("class ResizeAndPad-__call__: pad detail: top: {}, bottom: {}, left: {}, right: {}".format(top, bottom, left, right))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.fillv, self.fillv, self.fillv)
        )

        # keep the result ratio and pad for output box process
        self.padding_values = (top, bottom, left, right)

        return img

class ScrfdFaceDetector(object):
    def __init__(self, int=False, conf_thr=0.5, iou_thr=0.9, debug=False):
        """Face detector include 5pts landmark model

        Args:
            int (bool): use tflite(int) or tflite(float)
            conf_thr (float): confidence threshold, lower then more predict box
            iou_thr (float): iou threshold, lower then less predict box
            debug (bool): show the input/output details or not
        """
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.num_anchors = 2
        # load tflite model
        if int:
            #self.model_path = os.path.join("models", "scrfd500m_128x160_int8.tflite")  # 128 x 160
            #self.model_path = os.path.join("models", "scrfd500m_256x320_int8.tflite")  # 256 x 320
            self.model_path = os.path.join("models", "scrfd500m_480x640_int8.tflite")  # 480 x 640
        else:
            #self.model_path = os.path.join("models", "scrfd500m_128x160_float32.tflite")  # 128 x 160
            #self.model_path = os.path.join("models", "scrfd500m_256x320_float32.tflite")  # 256 x 320
            self.model_path = os.path.join("models", "scrfd500m_480x640_float32.tflite")  # 480 x 640
        print("-----[scrfd]read model from: " + self.model_path)
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if debug:
            print("---[scrfd]input details")
            for in_idx, in_dict in enumerate(self.input_details):
                print("input_{}, shape: {}".format(in_idx, in_dict['shape']))
            print("---[scrfd]output details")
            for out_idx, out_dict in enumerate(self.output_details):
                print("output_{}, shape: {}".format(out_idx, out_dict['shape']))

        # get tflite model input shape
        netin_shape = self.input_details[0]['shape']  # ndarray with shape (1, netin_height, netin_width, netin_c)
        print("-----[scrfd]netin_shape: {}".format(netin_shape))
        self.netin_height = netin_shape[1]
        self.netin_width = netin_shape[2]
        self.netin_shape = [self.netin_height, self.netin_width, netin_shape[3]]

        self.adjuster = ResizeAndPad(self.netin_shape, center=True, fillv=114, debug=debug)
    
    def adjust_input(self, cam_in):
        """Adjusts the input image(camera-in) to match the expected input size and format of the model.
        Resize cam_in to netin_shape

        Args:
            cam_in (ndarray): usually in the shape (720, 1280, 3) or (480, 640, 3)

        Returns:
            netin_image (ndarray): usually in the shape (128, 160, 3) or (256, 320, 3), can be visualize.
        """
        return self.adjuster(cam_in)

    def preprocess_image(self, image, auto_resize=False, BGR2RGB=True):
        """Preprocesses the input image for inference. include: BGR2RGB, normalize, and expand_dim

        Args:
            image (ndarray): usually cames from adjust_input
            auto_resize (bool): if image.shape != netin.shape, auto resize the input to the netin shape, WARNING, aspect ratio may not keep.
            BGR2RGB (bool): convert image from bgr to rgb channel

        Returns:
            netin (ndarray): with shape (batch=1, netin_height, netin_width, netin_c)
        """
        self.source_height, self.source_width, _ = image.shape
        if (self.source_height == self.netin_height) and (self.source_width == self.source_width):
            rs_image = image.copy()
        else:
            print("-----preprocess_image: get the unmatching image shape, process resize in the preprocess_image, aspect ratio may not keep")
            raise("unimplement auto_resize")
            rs_image = cv2.resize(image, (self.netin_width, self.netin_height))

        if BGR2RGB:
            image_rgb = cv2.cvtColor(rs_image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image.copy()
        # normalize
        netin = (image_rgb - 127.5) / 128
        netin = np.expand_dims(netin, 0).astype(np.float32)
        return netin

    def _invoke(self, netin):
        """Pure model inference

        Args:
            netin (ndarray): normalized image with batch shape (1, netin_height, netin_width, 3), converted to RGB input(From BGR)

        Returns:
            netout (dict): A dictonary contain face size string(['s', 'm', 'l']) as key, and their corresponding [conf, bbox, lms] as value.

        Note:
            Sometimes tflite model has different output node order should fix
        """
        self.interpreter.set_tensor(self.input_details[0]['index'], netin)
        self.interpreter.invoke()
        # output shape using [128, 160, 3] netin shape as example
        m_conf = self.interpreter.get_tensor(self.output_details[0]['index'])  # [160,   1]
        s_lm = self.interpreter.get_tensor(self.output_details[1]['index'])    # [640,  10]
        l_conf = self.interpreter.get_tensor(self.output_details[2]['index'])  # [40,    1]
        m_lm = self.interpreter.get_tensor(self.output_details[3]['index'])    # [160,  10]
        m_box = self.interpreter.get_tensor(self.output_details[4]['index'])   # [160,   4]
        l_box = self.interpreter.get_tensor(self.output_details[5]['index'])   # [40,    4]
        l_lm = self.interpreter.get_tensor(self.output_details[6]['index'])    # [40,   10]
        s_box = self.interpreter.get_tensor(self.output_details[7]['index'])   # [640,   4]
        s_conf = self.interpreter.get_tensor(self.output_details[8]['index'])  # [640,   1]

        return {'s': [s_conf, s_box, s_lm],
                'm': [m_conf, m_box, m_lm],
                'l': [l_conf, l_box, l_lm]
                }

    def postprocess_output(self, netout, top_k=None, debug=False):
        """Decode the netout to bbox and lms

        Args:
            netout (dict): A dictonary contain face size string(['s', 'm', 'l']) as key, and their corresponding [conf, bbox, lms] as value.
            debug (bool): for each stride, print number of detected box[before NMS] or not.

        Returns:
            bboxes (ndarray): with shape (N, 5), indicate [xmin, ymin, xmax, ymax, conf], where N is number of predict box [after NMS]
            lms (ndarray): with shape (N, 5, 2), indicate 5 pts landmark in the [x, y] order

        Note:
            we suppose batch size always be 1
        """
        scores_list = []
        bboxes_list = []
        kpss_list = []
        for box_size_str, stride in zip(['s', 'm', 'l'], [8, 16, 32]):
            anchor_centers = self.get_anchor_centers(self.netin_shape, stride, self.num_anchors)  # (num_grid * num_anchor, 2) for x(w) and y(h)
            conf = netout[box_size_str][0]          # (num_grid * num_anchor, 1)
            box = netout[box_size_str][1] * stride  # (num_grid * num_anchor, 4)
            kps = netout[box_size_str][2] * stride  # (num_grid * num_anchor, 10)

            pos_idx = np.where(conf >= self.conf_thr)[0]

            bboxes = self.distance2bbox(anchor_centers, box)  # (num_grid * num_anchor, 4)
            kpss = self.distance2kps(anchor_centers, kps)     # (num_grid * num_anchor, 10)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))       # (num_grid * num_anchor, 5, 2)
            
            pos_scores = conf[pos_idx]    # (num_det, 1)
            pos_bboxes = bboxes[pos_idx]  # (num_det, 4)
            pos_kpss = kpss[pos_idx]      # (num_det, 5, 2)

            if debug:
                print("stride: {} num_det before nms: {}".format(stride, len(pos_idx)))

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss_list.append(pos_kpss)
        if debug:
            print("---score list from postprocess_output: ")
            pprint(scores_list)
        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)
        # nms
        if debug:
            print("[scrfd.postprocess_output]: before-nms, bboxes has shape: {}".format(bboxes.shape))
            print("-------check nms input")
            print("--bboxes has shape: {}".format(bboxes.shape))  # (num_det, 4)
            print("--kpss has shape: {}".format(kpss.shape))  # (num_det, 5, 2)
            print("--scores.tolist(): {}".format(scores.tolist()))
            print("--conf_thr: {}".format(self.conf_thr))
            print("--iou_thr: {}".format(self.iou_thr))
        if bboxes.shape[0] == 0:
            print("no face det: early return empty ndarray")
            return np.empty((0, 4), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)
        keep = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.conf_thr, self.iou_thr, top_k=top_k)
        if debug:
            print("[scrfd.postprocess_output]: keep: {}".format(keep))
        if isinstance(keep, tuple):
            print("WARNING: quantized error occur, select first box automatically")
            keep = np.array([0])

        scores_keep = scores[keep]
        bboxes_keep = bboxes[keep]
        kpss_keep = kpss[keep]
        scores_keep = np.expand_dims(scores_keep, 1)
        bboxes = np.concatenate((bboxes_keep, scores_keep), axis=1)
        if debug:
            print("bboxes after nms has shape: {}".format(bboxes.shape))
            print("lms after nms has shape: {}".format(kpss_keep.shape))
        return bboxes, kpss_keep
    
    def _inference(self, netin_image, top_k=None, debug=False):
        """Given image with net-in shape, _inference output the corresponding box directly.
        
        Args:
            netin_image (ndarray): with shape (netin_height, netin_width, netin_c)
            debug (bool): print invoke time or not

        Returns:
            boxes (ndarray): with shape (num_pred, 5) in the [xmin, ymin, xmax, ymax, conf] order.
            lms (ndarray): with shape (num_pred, 5, 2), indicate 5 pts landmark in the [x, y] order
        """
        start = time.time()
        netin = self.preprocess_image(netin_image)
        invoke_start = time.time()
        netout = self._invoke(netin)
        invoke_end = time.time()
        boxes, lms = self.postprocess_output(netout, top_k=top_k, debug=debug)
        end = time.time()
        if debug:
            print("pure model invoke time: {:.4f}".format(invoke_end - invoke_start))
            print("single image inference time include (inner) pre/post process: {:.4f}".format(end - start))
        return boxes, lms

    def adjust_output_scale(self, bboxes, lms, debug=False):
        """Scale and shift the output box and landmarks to cam-in scale(original in the netin scale)

        Args:
            bboxes (ndarray): in [netin scale] with shape (num_pred, 5) in the [xmin, ymin, xmax, ymax, conf] order.
            lms (ndarray): in [netin scale] with shape (num_pred, 5, 2), indicate 5 pts landmark in the [x, y] order.
        
        Returns:
            bboxes (ndarray): in [cam-in scale] with shape (num_pred, 5) in the [xmin, ymin, xmax, ymax, conf] order.
            lms (ndarray): in [cam-in scale] with shape (num_pred, 5, 2), indicate 5 pts landmark in the [x, y] order.

        """
        ratio = self.adjuster.r
        (top, bottom, left, right) = self.adjuster.padding_values
        if debug:
            print("ratio from processor: {:.2f}".format(ratio))
            print("pad value: t: {}, b: {}, l: {}, r: {}".format(top, bottom, left, right))
        unpad_x = left
        unpad_y = top
        bboxes[:, 0] -= unpad_x  # xmin
        bboxes[:, 2] -= unpad_x  # xmax
        bboxes[:, 1] -= unpad_y  # ymin
        bboxes[:, 3] -= unpad_y  # ymax
        lms[:, :, 0] -= unpad_x
        lms[:, :, 1] -= unpad_y
        un_ratio = 1 / ratio
        bboxes[:, 0:4] *= un_ratio
        lms*= un_ratio
        return bboxes, lms


    def inference(self, image, top_k=None, debug=False):
        """Full pipeline scrfd inference from image to bbox and lms.
        
        Args:
            image (ndarray): camera-in usually in the shape (720, 1280, 3) or (480, 640), BGR order
        
        Returns:
            boxes (ndarray): in [cam-in scale] with shape (num_pred, 5) in the [xmin, ymin, xmax, ymax, conf] order.
            lms (ndarray): in [cam-in scale] with shape (num_pred, 5, 2), indicate 5 pts landmark in the [x, y] order.
        """
        netin_image = self.adjust_input(image)
        if debug:
            print("[scrfd.inference]: netin_image has shape: {}".format(netin_image.shape))
        boxes, lms = self._inference(netin_image, debug=debug)
        if debug:
            print("[scrfd.inference]: _inference: \n    boxes shape: {}, lms shape: {}".format(boxes.shape, lms.shape))
        boxes, lms = self.adjust_output_scale(boxes, lms, debug=debug)
        if debug:
            print("[scrfd.inference]: after adjust_output_scale: \n     boxes has shape: {}, lms has shape: {}".format(boxes.shape, lms.shape))
        return boxes, lms

    def distance2bbox(self, points, distance, max_shape=None):
        """Helper function for decode-netout (inner-post-processing), adjust the bbox to [xmin, ymin, xmax, ymax]
        Args:
            points (ndarray): anchor center in pix coord.(scale)
            distance (ndarray): bbox_preds(netout)
            max_shape (int/None): clip the predict box to (0, max_shape)

        Returns:
            bboxes (ndarray): with shape (num_pred, 4) in the [xmin, ymin, xmax, ymax] order
        """
        x1 = points[:, 0] - distance[:, 0]  # xmin
        y1 = points[:, 1] - distance[:, 1]  # ymin
        x2 = points[:, 0] + distance[:, 2]  # xmax
        y2 = points[:, 1] + distance[:, 3]  # ymax
        if max_shape is not None:
            x1 = x1.clip(min=0, max=max_shape[1])
            y1 = y1.clip(min=0, max=max_shape[0])
            x2 = x2.clip(min=0, max=max_shape[1])
            y2 = y2.clip(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance, max_shape=None):
        """Helper function for decode-netout (inner-post-processing), adjust the landmarks
        
        Args:
            points (ndarray): anchor center in pix coord
            distance (ndarray): kpss pred(netout)
            max_shape (int/None): clip the predict box to (0, max_shape)

        Returns:
            lms (ndarray): in [netin scale] with shape (num_pred, 5, 2), indicate 5 pts landmark in the [x, y] order.
        """
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clip(min=0, max=max_shape[1])
                py = py.clip(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def get_anchor_centers(self, netin_shape, stride, num_anchors):
        """Get anchor center for each output feature map

        Args:
            netin_shape (tuple): (netin_height, netin_width)
            stride (int): usually in the [8, 16, 32]
            num_anchors (int): 2 in scrfd500m models.

        Returns:
            anchor_centers (ndarray): with shape (netin_shape//stride, netin_shape//stride, 2 * num_anchors).reshape(-1, num_anchors)
                where, the magic number 2 above corresponding to (width, and height)
        """
        netin_h = netin_shape[0]
        netin_w = netin_shape[1]
        grid_h = netin_h // stride
        grid_w = netin_w // stride
        anchor_centers = np.stack(np.mgrid[:grid_h, :grid_w][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
        return anchor_centers

    def plot(self,
             im2show,
             boxes,
             kps,
             color=(0, 255, 0),
             num=False):
        """Plot boxes, conf, kps

        Args:
            im2show (ndarray): image to visualized
            boxes (ndarray): with shape (N, 5), indicate [xmin, ymin, xmax, ymax, conf]
            kps (ndarray): with shape (N, 5, 2), indicate landmarks [ptx, pty] in the order [right_eyes, left_eyes, nose, right_mouth, left_mouth]
            color (tuple): 
            num (bool): show landmark index or not
        """
        if boxes.shape[0] == 0:
            print("plot: no face det")
            return
        for box_idx, (box, kp) in enumerate(zip(boxes, kps)):
            x1, y1, x2, y2, conf = box
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            cv2.rectangle(im2show, (x1, y1), (x2, y2), color , thickness=2)
            cv2.putText(im2show, str(round(conf, 2)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=2)
            for lm_idx in range(5):
                cv2.circle(im2show, (int(kp[lm_idx, 0]), int(kp[lm_idx, 1])), 2, color, thickness=-1)
                if num:
                    cv2.putText(im2show, str(lm_idx), (int(kp[lm_idx, 0]), int(kp[lm_idx, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=2)

if __name__ == "__main__":
    # Simple usage example:
    # Except usage: python scrfd.py
    ## model initial
    fd = ScrfdFaceDetector(int=False, conf_thr=0.5, iou_thr=0.3, debug=True)
    ## data initial
    image_path = os.path.join("data", "example", "friends.jpg")
    image = cv2.imread(image_path)
    im2show = image.copy()

    # infernece
    bboxes, lms = fd.inference(image, debug=True)
    fd.plot(im2show, bboxes, lms, num=False)

    cv2.imshow("test", im2show)
    key = cv2.waitKey(0)
    if key == ord('s'):
        image_name = os.path.basename(image_path).strip(".jpg")
        cv2.imwrite(os.path.join("results", image_name + "_result.jpg"), im2show)
    cv2.destroyAllWindows()