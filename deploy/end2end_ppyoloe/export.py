import argparse
from pathlib import Path
from collections import OrderedDict

import onnx
import numpy as np
import onnx_graphsurgeon as gs


def export_onnx(opt: argparse.Namespace) -> str:
    # Ensure the required modules are imported within the function scope
    from paddle2onnx.legacy.command import program2onnx

    model_dir = Path(opt.model_dir)
    save_file = Path(opt.save_file)

    # Validate model directory
    assert model_dir.exists() and model_dir.is_dir(), f"Invalid model directory: {model_dir}"

    # Adjust save_file if it's a directory or has an incorrect suffix
    if save_file.is_dir():
        save_file /= model_dir.stem
    if save_file.suffix != '.onnx':
        save_file = save_file.with_suffix('.onnx')
    save_file = str(save_file)

    # Define input shape dictionary
    input_shape_dict = {
        'image': [opt.batch_size, 3, *opt.imgsz], 
        'scale_factor': [opt.batch_size, 2]
    }

    # Export the model to ONNX
    program2onnx(
        model_dir=opt.model_dir,
        save_file=save_file,
        model_filename=opt.model_filename,
        params_filename=opt.params_filename,
        opset_version=opt.opset,
        input_shape_dict=input_shape_dict
    )

    return save_file


def efficient_nms_plugin(opt: argparse.Namespace, model_path: str) -> None:
    # Load the ONNX model and simplify it
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    try:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, 'Simplified ONNX model could not be validated.'
    except Exception as e:
        print(f'Simplifier failure: {e}')

    # Import the ONNX model into Graph Surgeon
    graph = gs.import_onnx(model).fold_constants().cleanup().toposort()

    # Find Mul node
    mul_node = next((node.i(0) for node in graph.nodes if node.op == 'Div' and node.i(0).op == 'Mul'), None)

    # Find Concat node
    concat_node = next((node for node in graph.nodes if node.op == 'Concat' and len(node.inputs) == 3
                        and all(node.i(idx).op == 'Reshape' for idx in range(3))), None)

    # Ensure Mul and Concat nodes are found
    assert mul_node is not None, "Mul node not found."
    assert concat_node is not None, "Concat node not found."

    # Extract relevant information from nodes
    anchors = int(mul_node.inputs[1].shape[0])
    classes = int(concat_node.i(0).inputs[1].values[1])
    sum_anchors = int(concat_node.i(0).inputs[1].values[2] + concat_node.i(1).inputs[1].values[2] + concat_node.i(2).inputs[1].values[2])

    # Check equality condition
    assert anchors == sum_anchors, f"{mul_node.inputs[1].name}.shape[0] must equal the sum of values[2] from the three Concat nodes."

    # Create a new variable for 'scores' and transpose it
    scores = gs.Variable(name='scores', shape=[opt.batch_size, anchors, classes], dtype=np.float32)
    graph.layer(op='Transpose', name='last.Transpose',
                inputs=[concat_node.outputs[0]],
                outputs=[scores],
                attrs=OrderedDict(perm=[0, 2, 1]))
    graph.inputs = [graph.inputs[0]]

    # Define attributes for EfficientNMS
    attrs = OrderedDict(
        plugin_version="1",
        background_class=-1,
        max_output_boxes=opt.max_boxes,
        score_threshold=opt.conf_thres,
        iou_threshold=opt.iou_thres,
        score_activation=False,
        box_coding=0
    )

    # Define output variables
    outputs = [
        gs.Variable("num_detections", np.int32, [opt.batch_size, 1]),
        gs.Variable("detection_boxes", np.float32, [opt.batch_size, opt.max_boxes, 4]),
        gs.Variable("detection_scores", np.float32, [opt.batch_size, opt.max_boxes]),
        gs.Variable("detection_classes", np.int32, [opt.batch_size, opt.max_boxes])
    ]

    # Add EfficientNMS layer to the graph
    graph.layer(op='EfficientNMS_TRT', name="batched_nms",
                inputs=[mul_node.outputs[0], scores],
                outputs=outputs,
                attrs=attrs)

    # Update graph outputs
    graph.outputs = outputs
    graph.cleanup().toposort()

    # Save the modified ONNX model
    onnx.save(gs.export_onnx(graph), model_path)
    print("Finished!")


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='Path of directory saved the input model.')
    parser.add_argument('--model_filename', required=True, type=str, help='The input model file name.')
    parser.add_argument('--params_filename', required=True, type=str, help='The parameters file name.')
    parser.add_argument('--save_file', required=True, type=str, help='Path of directory to save the new exported model.')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for model.')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold.')
    parser.add_argument('--opset', type=int, default=11, help='opset version.')
    parser.add_argument('--max_boxes', type=int, default=100, help='The maximum number of detections to output per image.')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    model_path = export_onnx(opt)
    efficient_nms_plugin(opt, model_path)
