# Modified from [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)

import os
import pickle

from .clip import *
from ppdet.utils.download import get_weights_path

COCO_CATEGORIES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "street sign",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eye glasses",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    45: "plate",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    66: "mirror",
    67: "dining table",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "computer mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
    91: "hair brush",
}


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


single_template = ["a photo of a {}."]

multiple_templates = [
    "There is {article} {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {article} {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of {article} {}.",
    "itap of my {}.",  # itap: I took a picture of
    "itap of the {}.",
    "a photo of {article} {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {article} {}.",
    "a good photo of the {}.",
    "a bad photo of {article} {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {article} {}.",
    "a bright photo of the {}.",
    "a dark photo of {article} {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {article} {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {article} {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {article} {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {article} {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {article} {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {article} {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {article} {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]


def load_model(model_name, clip_path, pretrained=False):
    model_fn, url, file_name = model_dict[model_name]
    model, transforms = model_fn()
    model_path = os.path.join(clip_path, file_name)
    if pretrained:
        if not os.path.isfile(model_path):
            path = get_weights_path(url)
            model_path = path
            # if not os.path.exists('pretrained_models'):
            #     os.mkdir('pretrained_models')
            # wget.download(url, out=model_path)
        params = paddle.load(model_path)
        res = match_state_dict(model.state_dict(), params)
        model.set_state_dict(params)
    model.eval()
    return model, transforms


def load_clip_to_cpu(visual_backbone, clip_path):
    model, _ = load_model(visual_backbone, clip_path, pretrained=True)
    return model


class TextEncoder(nn.Layer):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        # self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    @paddle.no_grad()
    def forward(self, text):
        x = self.token_embedding(text).astype(
            'float32')  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.astype('float32')
        x = self.transformer(x)
        x = self.ln_final(x).astype('float32')

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        batch_idx = paddle.arange(x.shape[0])
        seq_idx = text.argmax(axis=-1)
        gather_idx = paddle.stack([batch_idx, seq_idx], axis=1)
        x = paddle.gather_nd(x, gather_idx)
        x = x @self.text_projection

        return x


def build_text_embedding_coco(bpe_path, clip_path):
    categories = COCO_CATEGORIES
    run_on_gpu = True

    clip_model = load_clip_to_cpu("ViT_B_32", clip_path)
    text_model = TextEncoder(clip_model)

    for name, param in text_model.named_parameters():
        param.stop_gradient = True
    templates = multiple_templates
    with paddle.no_grad():
        zeroshot_weights = []
        for _, category in categories.items():
            texts = [
                template.format(
                    processed_name(
                        category, rm_dot=True),
                    article=article(category)) for template in templates
            ]
            texts = [
                "This is " + text
                if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]
            texts = tokenize(texts, bpe_path=bpe_path)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = text_model(texts)
            text_embeddings /= paddle.linalg.norm(
                text_embeddings, axis=-1, keepdim=True)
            text_embedding = paddle.mean(text_embeddings, axis=0)
            text_embedding /= paddle.linalg.norm(text_embedding)
            zeroshot_weights.append(text_embedding)
        zeroshot_weights = paddle.stack(zeroshot_weights, axis=1)
        if run_on_gpu:
            zeroshot_weights = zeroshot_weights.cuda()
    zeroshot_weights = zeroshot_weights.t().numpy()
    all_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        27, 28, 31, 32, 33, 34, 35, 36, 38, 41, 42, 44, 47, 48, 49, 50, 51, 52,
        53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 70, 72, 73, 74, 75, 76, 78,
        79, 80, 81, 82, 84, 85, 86, 87, 90
    ]  # noqa
    all_ids = [i - 1 for i in all_ids]

    return paddle.to_tensor(zeroshot_weights[all_ids])


def read_clip_feat(clip_feat_path):
    url = 'https://bj.bcebos.com/v1/paddledet/data/coco/clip_feat_coco_pickle_label.pkl'
    if not os.path.exists(clip_feat_path):
        path = os.path.expanduser("~/.cache/paddle/weights")
        # path = get_weights_path(url)
        clip_feat_path = os.path.join(path, url.split('/')[-1])
        if not os.path.exists(clip_feat_path):
            wget.download(url, clip_feat_path)

    with open(clip_feat_path, 'rb') as f:
        clip_feat = pickle.load(f)
        return clip_feat
