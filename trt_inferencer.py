from __future__ import annotations
import os
import sys
import time
import common
import pycuda.driver as cuda
from pathlib import Path
import tensorrt as trt
from typing import Any, cast
import albumentations as A
import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf
from anomalib.data import TaskType
from anomalib.data.utils import read_image
from anomalib.post_processing import ImageResult
from anomalib.post_processing.normalization.cdf import normalize as normalize_cdf
from anomalib.post_processing.normalization.cdf import standardize
from anomalib.post_processing.normalization.min_max import normalize as normalize_min_max
from anomalib.post_processing import Visualizer

TRT_LOGGER = trt.Logger()

class TrtInferencer():
    """TensorRT implementation for the inference.
    """
    def __init__(
        self,
        path,
        metadata,
        batch_size=1,
        dynamic_batch=True,
        # device="CPU",
        task=None,
        config=None,
    ):
        # self.device = device
        self.config = config
        # self.input_blob, self.output_blob, self.model = self.load_model(path)
        self.metadata = self.load_metadata(metadata)
        self.batch_size = batch_size

        print('inference batchsize = ' + str(self.batch_size))

        self.dynamic_batch = dynamic_batch
        self.task = TaskType(task) if task else TaskType(self.metadata["task"])

        self.input_h = self.metadata["transform"]["transform"]["transforms"][0]["height"]  ######################
        self.input_w = self.metadata["transform"]["transform"]["transforms"][0]["width"]

        # if self.batch_size > 1:
        self.metadata["image_shape_batch"] = list([None for i in range(self.batch_size)])

        if self.task == TaskType.CLASSIFICATION:
            self.output_shape = (-1, 1)
        elif self.task == TaskType.SEGMENTATION:
            self.output_shape = (-1, self.input_h, self.input_w)

        elif self.task == TaskType.DETECTION:
            print('Detection task is not supported yet!')
            sys.exit(1)

        self.engine = self.load_engine(path)
        self.context = self.engine.create_execution_context()

        # set input binding shape
        input_idx = self.engine.get_binding_index('input')
        if self.dynamic_batch:
            self.context.set_binding_shape(input_idx, (self.batch_size, 3, self.input_h, self.input_w))

        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine, self.batch_size)

        # warm up
        self.warm_up()

    def load_metadata(self, path):
        """Loads the meta data from the given path.

        Args:
            path (str | Path | dict | None, optional): Path to JSON file containing the metadata.
                If no path is provided, it returns an empty dict. Defaults to None.

        Returns:
            dict | DictConfig: Dictionary containing the metadata.
        """
        # metadata: dict[str, float | np.ndarray | Tensor] | DictConfig = {}
        print("Reading metadata from file {}...".format(path))
        metadata = DictConfig = {}
        if path is not None:
            config = OmegaConf.load(path)
            metadata = cast(DictConfig, config)

        print('metadata: ', metadata)
        return metadata

    def load_engine(self, engine_file_path):
        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}...".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            print("Engine file does not exist!")
            return None


    def _normalize(self, pred_scores, metadata, anomaly_maps=None):
        """Applies normalization and resizes the image.

        Args:
            pred_scores (Tensor | np.float32): Predicted anomaly score
            metadata (dict | DictConfig): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
            anomaly_maps (Tensor | np.ndarray | None): Predicted raw anomaly map.

        Returns:
            tuple[np.ndarray | Tensor | None, float]: Post processed predictions that are ready to be
                visualized and predicted scores.
        """

        # min max normalization
        if "min" in metadata and "max" in metadata:
            if anomaly_maps is not None:
                anomaly_maps = normalize_min_max(
                    anomaly_maps,
                    metadata["pixel_threshold"],
                    metadata["min"],
                    metadata["max"],
                )
            pred_scores = normalize_min_max(
                pred_scores,
                metadata["image_threshold"],
                metadata["min"],
                metadata["max"],
            )

        # standardize pixel scores
        if "pixel_mean" in metadata.keys() and "pixel_std" in metadata.keys():
            if anomaly_maps is not None:
                anomaly_maps = standardize(
                    anomaly_maps, metadata["pixel_mean"], metadata["pixel_std"], center_at=metadata["image_mean"]
                )
                anomaly_maps = normalize_cdf(anomaly_maps, metadata["pixel_threshold"])

        # standardize image scores
        if "image_mean" in metadata.keys() and "image_std" in metadata.keys():
            pred_scores = standardize(pred_scores, metadata["image_mean"], metadata["image_std"])
            pred_scores = normalize_cdf(pred_scores, metadata["image_threshold"])

        return anomaly_maps, float(pred_scores)

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        """Pre process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: pre-processed image.
        """
        transform = A.from_dict(self.metadata["transform"])
        processed_image = transform(image=image)["image"]

        if len(processed_image.shape) == 3:
            processed_image = np.expand_dims(processed_image, axis=0)

        if processed_image.shape[-1] == 3:
            processed_image = processed_image.transpose(0, 3, 1, 2)

        return processed_image

    def warm_up(self):
        image = np.zeros((3, self.input_h, self.input_w))
        image = np.expand_dims(image, axis=0)
        if self.batch_size > 1:
            image.repeat(self.batch_size-1, axis=0) # B, C, H, W
        image = np.ascontiguousarray(image, dtype=np.float32)
        self.inputs[0].host = image
        common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        print("warm up finished...")

    def predict(self, image, metadata=None):
        """Perform a prediction for a given input image.

        The main workflow is (i) pre-processing, (ii) forward-pass, (iii) post-process.

        Args:
            image (np.ndarray): Input image whose output is to be predicted.

            metadata: Metadata information such as shape, threshold.

        Returns:
            ImageResult: Prediction results to be visualized.
        """
        if metadata is None:
            if hasattr(self, "metadata"):
                metadata = getattr(self, "metadata")
            else:
                metadata = {}

        # if isinstance(image, (str, Path)):
        #     image_arr: np.ndarray = read_image(image)
        # else:  # image is already a numpy array. Kept for mypy compatibility.
        #     image_arr = image

        image_arr = image
        metadata["image_shape"] = image_arr.shape[:2]

        processed_image = self.pre_process(image_arr)
        processed_image = np.ascontiguousarray(processed_image, dtype=np.float32) ##################################### B, C, H, W

        self.inputs[0].host = processed_image
        trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        # trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]

        # predictions = trt_outputs[-1] ##########################

        predictions = trt_outputs[-1].reshape(self.output_shape)

        output = self.post_process(predictions, metadata=metadata)

        print('images[i].shape:',image_arr.shape, image_arr.dtype)
        print('output["anomaly_map"].shape:',output["anomaly_map"].shape, output["anomaly_map"].dtype)

        return ImageResult(
            image=image_arr,
            pred_score=output["pred_score"],
            pred_label=output["pred_label"],
            anomaly_map=output["anomaly_map"],
            pred_mask=output["pred_mask"],
            pred_boxes=output["pred_boxes"],
            box_labels=output["box_labels"],
        )

    def predict_batch(self, images, metadata=None):
        """Perform a prediction for a given input image.

        The main workflow is (i) pre-processing, (ii) forward-pass, (iii) post-process.

        Args:
            images (List of np.ndarray): Input images whose output is to be predicted.

            metadata: Metadata information such as shape, threshold.

        Returns:
            ret: ImageResults(>=1) to be visualized.
        """
        if metadata is None:
            if hasattr(self, "metadata"):
                metadata = getattr(self, "metadata")
            else:
                metadata = {}
        # if isinstance(image, (str, Path)):
        #     image_arr: np.ndarray = read_image(image)
        # else:  # image is already a numpy array. Kept for mypy compatibility.
        #     image_arr = image
        t_preprocess = time.time()

        processed_images = []
        for idx, image in enumerate(images):

            image_arr = image
            metadata["image_shape_batch"][idx] = image_arr.shape[:2]

            processed_image = self.pre_process(image_arr)
            processed_images.append(processed_image)
        # concat
        processed_images = np.concatenate(processed_images, axis=0)
        processed_images = np.ascontiguousarray(processed_images, dtype=np.float32) ##################################### B, C, H, W
        # print('processed_images.shape: ', processed_images.shape)
        print('preprocess time:', time.time()-t_preprocess)

        t_infer = time.time()

        self.inputs[0].host = processed_images
        trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        # print('trt_outputs', trt_outputs)
        # trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]

        predictions = trt_outputs[-1]
        predictions = predictions.reshape(self.output_shape)

        print('inference time:', time.time()-t_infer)

        t_postprocess = time.time()

        ret = []
        for i in range(self.batch_size):
            # print('predictions[i].shape', predictions[i].shape)
            if self.task == TaskType.SEGMENTATION:
                output = self.post_process_batch(i, np.expand_dims(predictions[i], axis=0), metadata=metadata)
            elif self.task == TaskType.CLASSIFICATION:
                output = self.post_process_batch(i, predictions[i], metadata=metadata)
            # print('images[i].shape:',images[i].shape, images[i].dtype)
            # print('output["anomaly_map"].shape:',output["anomaly_map"].shape, output["anomaly_map"].dtype)
            ret.append(ImageResult(
            image=images[i],
            pred_score=output["pred_score"],
            pred_label=output["pred_label"],
            anomaly_map=output["anomaly_map"],
            pred_mask=output["pred_mask"],
            pred_boxes=output["pred_boxes"],
            box_labels=output["box_labels"],
        ))

        print('postprocess time:', time.time()-t_postprocess)

        return ret

    def post_process(self, predictions: np.ndarray, metadata: dict | DictConfig | None = None) -> dict[str, Any]:
        """Post process the output predictions.

        Args:
            predictions (np.ndarray): Raw output predicted by the model.
            metadata (Dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to None.

        Returns:
            dict[str, Any]: Post processed prediction results.
        """
        if metadata is None:
            metadata = self.metadata

        # predictions = predictions[self.output_blob] #############################

        # Initialize the result variables.
        anomaly_map: np.ndarray | None = None
        pred_label: float | None = None
        pred_mask: float | None = None

        # If predictions returns a single value, this means that the task is
        # classification, and the value is the classification prediction score.
        if len(predictions.shape) == 1:
            task = TaskType.CLASSIFICATION
            pred_score = predictions
        else:
            task = TaskType.SEGMENTATION
            anomaly_map = predictions.squeeze()
            pred_score = anomaly_map.reshape(-1).max()

        # Common practice in anomaly detection is to assign anomalous
        # label to the prediction if the prediction score is greater
        # than the image threshold.
        # print('pred_score:', pred_score)
        if "image_threshold" in metadata:
            pred_label = pred_score >= metadata["image_threshold"]

        if task == TaskType.CLASSIFICATION:
            _, pred_score = self._normalize(pred_scores=pred_score, metadata=metadata)
        elif task in (TaskType.SEGMENTATION, TaskType.DETECTION):
            if "pixel_threshold" in metadata:
                pred_mask = (anomaly_map >= metadata["pixel_threshold"]).astype(np.uint8)

            anomaly_map, pred_score = self._normalize(
                pred_scores=pred_score, anomaly_maps=anomaly_map, metadata=metadata
            )
            assert anomaly_map is not None

            if "image_shape" in metadata and anomaly_map.shape != metadata["image_shape"]:
                image_height = metadata["image_shape"][0]
                image_width = metadata["image_shape"][1]
                anomaly_map = cv2.resize(anomaly_map, (image_width, image_height))

                if pred_mask is not None:
                    pred_mask = cv2.resize(pred_mask, (image_width, image_height))
        else:
            raise ValueError(f"Unknown task type: {task}")

        if self.task == TaskType.DETECTION:
            pred_boxes = self._get_boxes(pred_mask)
            box_labels = np.ones(pred_boxes.shape[0])
        else:
            pred_boxes = None
            box_labels = None

        return {
            "anomaly_map": anomaly_map,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "pred_mask": pred_mask,
            "pred_boxes": pred_boxes,
            "box_labels": box_labels,
        }

    def post_process_batch(self, img_index, predictions: np.ndarray, metadata: dict | DictConfig | None = None) -> dict[str, Any]:
        """Post process the output predictions.

        Args:
            img_index (int): Index of image in a batch.
            predictions (np.ndarray): Raw output predicted by the model.
            metadata (Dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to None.

        Returns:
            dict[str, Any]: Post processed prediction results.
        """
        if metadata is None:
            metadata = self.metadata

        # Initialize the result variables.
        anomaly_map: np.ndarray | None = None
        pred_label: float | None = None
        pred_mask: float | None = None

        # If predictions returns a single value, this means that the task is
        # classification, and the value is the classification prediction score.
        if len(predictions.shape) == 1:
            task = TaskType.CLASSIFICATION
            pred_score = predictions
        else:
            task = TaskType.SEGMENTATION
            anomaly_map = predictions.squeeze()
            pred_score = anomaly_map.reshape(-1).max()

        # Common practice in anomaly detection is to assign anomalous
        # label to the prediction if the prediction score is greater
        # than the image threshold.
        # print('pred_score:', pred_score)
        if "image_threshold" in metadata:
            pred_label = pred_score >= metadata["image_threshold"]

        if task == TaskType.CLASSIFICATION:
            _, pred_score = self._normalize(pred_scores=pred_score, metadata=metadata)
        elif task in (TaskType.SEGMENTATION, TaskType.DETECTION):
            if "pixel_threshold" in metadata:
                pred_mask = (anomaly_map >= metadata["pixel_threshold"]).astype(np.uint8)

            anomaly_map, pred_score = self._normalize(
                pred_scores=pred_score, anomaly_maps=anomaly_map, metadata=metadata
            )
            assert anomaly_map is not None

            if "image_shape_batch" in metadata and anomaly_map.shape != metadata["image_shape_batch"]:
                image_height = metadata["image_shape_batch"][img_index][0]
                image_width = metadata["image_shape_batch"][img_index][1]
                anomaly_map = cv2.resize(anomaly_map, (image_width, image_height))

                if pred_mask is not None:
                    pred_mask = cv2.resize(pred_mask, (image_width, image_height))
        else:
            raise ValueError(f"Unknown task type: {task}")

        if self.task == TaskType.DETECTION:
            pred_boxes = self._get_boxes(pred_mask)
            box_labels = np.ones(pred_boxes.shape[0])
        else:
            pred_boxes = None
            box_labels = None

        return {
            "anomaly_map": anomaly_map,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "pred_mask": pred_mask,
            "pred_boxes": pred_boxes,
            "box_labels": box_labels,
        }

    def _get_boxes(self, mask):
        """Get bounding boxes from masks.

        Args:
            masks (np.ndarray): Input mask of shape (H, W)

        Returns:
            np.ndarray: array of shape (N, 4) containing the bounding box coordinates of the objects in the masks
            in xyxy format.
        """
        _, comps = cv2.connectedComponents(mask)

        labels = np.unique(comps)
        boxes = []
        for label in labels[labels != 0]:
            y_loc, x_loc = np.where(comps == label)
            boxes.append([np.min(x_loc), np.min(y_loc), np.max(x_loc), np.max(y_loc)])
        boxes = np.stack(boxes) if boxes else np.empty((0, 4))
        return boxes

if __name__ == "__main__":

    # trt_inferencer = TrtInferencer(path="weights/efficient_ad.engine", metadata="data/metadata.json", batch_size=1, dynamic_batch = True)
    # visualizer = Visualizer(mode='full', task='segmentation')
    # # image = read_image('test_images/000.jpg', (trt_inferencer.input_w, trt_inferencer.input_h))
    # image = read_image('test_images/000.jpg')
    # predictions = trt_inferencer.predict(image)
    # output = visualizer.visualize_image(predictions)
    # visualizer.show(title="Output Image", image=output)

    save_folder = 'result'
    batch_size = 4
    test_data_folder = 'D:/surface_defect_datasets/mvtec_anomaly_detection/transistor/test'
    img_list = []
    for img_class in list(Path(test_data_folder).iterdir()):
        child_dir = Path(test_data_folder).joinpath(img_class)
        if not child_dir.exists():
            Path(child_dir).mkdir(parents=True, exist_ok=True)
            print('create ' + str(child_dir))
        for img_name in list(child_dir.iterdir()):
            img_list.append(child_dir.joinpath(img_name))

    num_batch = len(img_list) // batch_size

    trt_inferencer = TrtInferencer(path="weights/efficient_ad.engine", metadata="data/metadata.json", batch_size=batch_size)
    visualizer = Visualizer(mode='full', task='segmentation')

    for batch_index in range(num_batch):
        print('batch ' + str(batch_index+1) + ' of ' + str(num_batch))
        images = []
        for i in range(batch_size):
            image_path_full = Path(img_list[batch_index*batch_size+i])
            images.append(read_image(image_path_full, (256, 256)))

        predictions = trt_inferencer.predict_batch(images)
        for i in range(batch_size):
            print(img_list[batch_index*batch_size+i])
            # print(predictions[i].pred_score)
            print('predict label:', predictions[i].pred_label) # True for anomalous

            # visualize
            output = visualizer.visualize_image(predictions[i])
            # visualizer.show(title="Output Image "+str(i+1), image=output)
            image_path_full = Path(img_list[batch_index*batch_size+i])
            path_parent = image_path_full.parent #
            dirname_class = path_parent.name
            image_name = image_path_full.name
            save_path_full = Path(save_folder).joinpath(dirname_class, image_name)
            visualizer.save(file_path=save_path_full, image=output)

