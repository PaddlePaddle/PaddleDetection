import argparse
import onnx
import onnx_graphsurgeon as gs
import numpy as np

from pathlib import Path
from paddle2onnx.legacy.command import program2onnx
from collections import OrderedDict


def main(opt):
    model_dir = Path(opt.model_dir)
    save_file = Path(opt.save_file)
    assert model_dir.exists() and model_dir.is_dir()
    if save_file.is_dir():
        save_file = (save_file / model_dir.stem).with_suffix('.onnx')
    elif save_file.is_file() and save_file.suffix != '.onnx':
        save_file = save_file.with_suffix('.onnx')
    input_shape_dict = {'image': [opt.batch_size, 3, *opt.img_size],
                        'scale_factor': [opt.batch_size, 2]}
    program2onnx(str(model_dir), str(save_file),
                 'model.pdmodel', 'model.pdiparams',
                 opt.opset, input_shape_dict=input_shape_dict)
    onnx_model = onnx.load(save_file)
    try:
        import onnxsim
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'assert check failed'
    except Exception as e:
        print(f'Simplifier failure: {e}')
    onnx.checker.check_model(onnx_model)
    graph = gs.import_onnx(onnx_model)
    graph.fold_constants()
    graph.cleanup().toposort()
    mul = concat = None
    for node in graph.nodes:
        if node.op == 'Div' and node.i(0).op == 'Mul':
            mul = node.i(0)
        if node.op == 'Concat' and node.o().op == 'Reshape' and node.o().o().op == 'ReduceSum':
            concat = node

    assert mul.outputs[0].shape[1] == concat.outputs[0].shape[2], 'Something wrong in outputs shape'

    anchors = mul.outputs[0].shape[1]
    classes = concat.outputs[0].shape[1]

    scores = gs.Variable(name='scores', shape=[opt.batch_size, anchors, classes], dtype=np.float32)
    graph.layer(op='Transpose', name='lastTranspose',
                inputs=[concat.outputs[0]],
                outputs=[scores],
                attrs=OrderedDict(perm=[0, 2, 1]))

    graph.inputs = [graph.inputs[0]]

    attrs = OrderedDict(
        plugin_version="1",
        background_class=-1,
        max_output_boxes=opt.topk_all,
        score_threshold=opt.conf_thres,
        iou_threshold=opt.iou_thres,
        score_activation=False,
        box_coding=0, )
    outputs = [gs.Variable("num_dets", np.int32, [opt.batch_size, 1]),
               gs.Variable("det_boxes", np.float32, [opt.batch_size, opt.topk_all, 4]),
               gs.Variable("det_scores", np.float32, [opt.batch_size, opt.topk_all]),
               gs.Variable("det_classes", np.int32, [opt.batch_size, opt.topk_all])]
    graph.layer(op='EfficientNMS_TRT', name="batched_nms",
                inputs=[mul.outputs[0], scores],
                outputs=outputs,
                attrs=attrs)
    graph.outputs = outputs
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), save_file)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str,
                        default=None,
                        help='paddle static model')
    parser.add_argument('--save-file', type=str,
                        default=None,
                        help='onnx model save path')
    parser.add_argument('--opset', type=int, default=11, help='opset version')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
