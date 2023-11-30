from argparse import ArgumentParser, Namespace
from pathlib import Path
from anomalib.data.utils import read_image
from anomalib.post_processing import Visualizer
from trt_inferencer import TrtInferencer
import numpy as np

def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=4, required=True, help="Inference batch size")
    parser.add_argument("--weights", type=Path, default=Path("weights/efficient_ad.engine"), required=True, help="Path to model weights")
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata_transistor_efficient_ad.json"), required=True, help="Path to a JSON file containing the metadata.")
    parser.add_argument("--input", type=Path, default=Path("D:/surface_defect_datasets/mvtec_anomaly_detection/transistor/test"), required=True, help="Path to images to infer.")
    parser.add_argument("--output", type=Path, default=Path("result"), required=True, help="Path to save the output images.")
    parser.add_argument("--visualize", type=int, default=1, required=True, help="Save the output images.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task type.",
        default="segmentation",
        choices=["classification", "detection", "segmentation"],
    )

    parser.add_argument(
        "--visualization_mode",
        type=str,
        required=True,
        default="full",
        help="Visualization mode.",
        choices=["full", "simple"],
    )
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        required=True,
        help="Show the visualized predictions on the screen.",
    )

    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()

    img_list = []
    print("Get images from folder {}...".format(args.input))
    if args.visualize:
        print("Save output images to folder {}...".format(args.output))
    for img_class in list(Path(args.input).iterdir()):
        child_dir = Path(args.input).joinpath(img_class)
        if not child_dir.exists():
            Path(child_dir).mkdir(parents=True, exist_ok=True)
            print('create ' + str(child_dir))
        for img_name in list(child_dir.iterdir()):
            img_list.append(child_dir.joinpath(img_name))

    num_batch = len(img_list) // args.batchsize

    trt_inferencer = TrtInferencer(path=args.weights, metadata=args.metadata, batch_size=args.batchsize)

    if args.visualize:
        visualizer = Visualizer(mode=args.visualization_mode, task=args.task)

    for batch_index in range(num_batch):
        print('batch ' + str(batch_index+1) + ' of ' + str(num_batch))
        images = []
        for i in range(args.batchsize):
            image_path_full = Path(img_list[batch_index*args.batchsize+i])
            # images.append(read_image(image_path_full, (256, 256))) # resize
            images.append(read_image(image_path_full))

        predictions = trt_inferencer.predict_batch(images)
        for i in range(args.batchsize):
            print(img_list[batch_index*args.batchsize+i])
            # print(predictions[i].pred_score)
            pred_label = predictions[i].pred_label
            pred_label = pred_label[0] if isinstance(pred_label, np.ndarray) or isinstance(pred_label, list) else pred_label
            print('predict label:', pred_label) # True for anomalous

            # visualize
            if args.visualize:
                output = visualizer.visualize_image(predictions[i])
                if args.show:
                    # show result
                    visualizer.show(title="Output Image "+str(batch_index*args.batchsize+i+1), image=output)
                # save result
                image_path_full = Path(img_list[batch_index*args.batchsize+i])
                path_parent = image_path_full.parent #
                dirname_class = path_parent.name
                image_name = image_path_full.name
                save_path_full = Path(args.output).joinpath(dirname_class, image_name)
                visualizer.save(file_path=save_path_full, image=output)

    trt_inferencer.destroy()
