from typing import List, Dict, Tuple, Any
import argparse
import os
import glob
import tensorrt as trt
import cupy as cp
import numpy as np
import cv2


def letterbox(im: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114),
              auto: bool = True, scaleFill: bool = False, scaleup: bool = True, stride: int = 32) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    height, width = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / height, new_shape[1] / width)
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(width * r)), int(round(height * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / width, new_shape[0] / height  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if (height, width) != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def visualize(image: np.ndarray, pred: Dict[str, Any], color: Tuple[int, int, int] = (128, 128, 128), txt_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # line width
    tf = max(lw - 1, 1)  # font thickness
    sf = lw / 3  # font scale

    vis_image = image.copy()
    for i in range(pred['num']):
        p1, p2 = (int(pred['bboxes'][i][0]), int(pred['bboxes'][i][1])), (int(pred['bboxes'][i][2]), int(pred['bboxes'][i][3]))
        cv2.rectangle(vis_image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

        label = f"cls: {pred['classes'][i]}, conf: {pred['scores'][i]:.2f}"
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(vis_image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(vis_image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    sf,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

    return vis_image


class PPYOLOE:

    def __init__(self, engine_path: str):
        self.stream = cp.cuda.Stream()  # Use cupy.cuda.Stream instead of driver.Stream
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace="")

        with open(engine_path, 'rb') as file, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(file.read())

        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []

        for binding in engine:
            shape = engine.get_binding_shape(binding)
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            device_array = cp.empty(shape, dtype)  # Use cupy.empty instead of driver.pagelocked_empty

            # Append the device buffer to device bindings.
            self.bindings.append(int(device_array.data))
            if engine.binding_is_input(binding):
                self.batch_size, _, self.width, self.height = engine.get_binding_shape(binding)
                self.inputs.append(device_array)
            else:
                self.outputs.append(device_array)

    def batch_infer(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        outputs = []
        for grouped_images in [images[i:i + self.batch_size] for i in range(0, len(images), self.batch_size)]:
            batch_image_ratio, batch_image_dwdh = [], []

            batch_input_image = cp.empty(shape=[self.batch_size, 3, self.width, self.height])
            for idx, image in enumerate(grouped_images):
                input_image, ratio, dwdh = self.preprocess(image)
                batch_image_ratio.append(ratio)
                batch_image_dwdh.append(dwdh)
                cp.copyto(batch_input_image[idx], input_image)  # Use cupy.copyto instead of np.copyto
            batch_input_image = cp.ascontiguousarray(batch_input_image)

            # Copy input image to host buffer
            cp.copyto(self.inputs[0], batch_input_image)

            # Run inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.ptr)

            # Synchronize stream
            self.stream.synchronize()

            outputs.extend(self.postprocess([cp.asnumpy(out) for out in self.outputs], batch_image_ratio, batch_image_dwdh))

        return outputs

    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        return self.batch_infer([image])[0]

    def preprocess(self, image: np.ndarray) -> Tuple[cp.ndarray, Tuple[float, float], Tuple[float, float]]:
        image, ratio, dwdh = letterbox(image, new_shape=(self.width, self.height), auto=False)
        image = cp.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(cp.float32) / 255.0
        image -= cp.array([0.485, 0.456, 0.406], dtype=cp.float32)[None, None, :]
        image /= cp.array([0.229, 0.224, 0.225], dtype=cp.float32)[None, None, :]
        image = image.transpose((2, 0, 1))
        return image, ratio, dwdh

    def postprocess(self, batch_output: List[np.ndarray], batch_image_ratio: List[Tuple[float, float]],
                    batch_image_dwdh: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        results = []
        nums = batch_output[0].reshape(self.batch_size)
        bboxes = batch_output[1].reshape(self.batch_size, 100, 4)
        scores = batch_output[2].reshape(self.batch_size, 100, 1)
        classes = batch_output[3].reshape(self.batch_size, 100, 1)

        for idx, ((rw, rh), (dw, dh)) in enumerate(zip(batch_image_ratio, batch_image_dwdh)):
            result = {
                'num': int(nums[idx]),
                'classes': [],
                'scores': [],
                'bboxes': [],
            }
            for i in range(int(nums[idx])):
                left = (bboxes[idx][i][0] - dw) / rw
                top = (bboxes[idx][i][1] - dh) / rh
                right = (bboxes[idx][i][2] - dw) / rw
                bottom = (bboxes[idx][i][3] - dh) / rh
                result['classes'].append(int(classes[idx][i]))
                result['scores'].append(float(scores[idx][i]))
                result['bboxes'].append([left, top, right, bottom])
            results.append(result)

        return results


def get_test_images(infer_dir: str, infer_img: str) -> List[str]:
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)

    return images


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True, type=str, help='Path to the TensorRT engine file.')
    parser.add_argument("--infer_dir", type=str, default=None, help="Directory for images to perform inference on.")
    parser.add_argument("--infer_img", type=str, default=None, help="Image path, has higher priority over --infer_dir")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for storing the output visualization files.")

    opt = parser.parse_args()
    return opt


def main():
    # Parse command line arguments
    opt = parse_opt()

    # Get paths to inference images
    images = get_test_images(opt.infer_dir, opt.infer_img)

    # Read original images
    ori_images = [cv2.imread(image) for image in images]

    # Create PPYOLOE model using the provided TensorRT engine
    model = PPYOLOE(opt.engine)

    # Perform batch inference on the original images
    results = model.batch_infer(ori_images)

    # Generate visualizations for the results
    vis_images = [visualize(image, result) for image, result in zip(ori_images, results)]

    # Create the output directory if it doesn't exist
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    # Save the visualizations to the output directory
    for image_path, vis_image in zip(images, vis_images):
        filename = os.path.basename(image_path)
        cv2.imwrite(os.path.join(opt.output_dir, filename), vis_image)

    # Print a message indicating where visualizations are saved
    print(f'Saved visualizations in {opt.output_dir}')


if __name__ == "__main__":
    main()
