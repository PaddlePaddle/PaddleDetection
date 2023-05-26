import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.modeling.initializer import constant_
from paddle.nn.initializer import KaimingNormal


class ConvModule(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 norm_type='bn',
                 wtih_act=True):
        super(ConvModule, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn', None]
        self.with_norm = norm_type is not None
        self.wtih_act = wtih_act
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias,
            weight_attr=KaimingNormal())
        if self.with_norm:
            if norm_type == 'bn':
                self.bn = nn.BatchNorm2D(out_channels)
            elif norm_type == 'gn':
                self.bn = nn.GroupNorm(out_channels, out_channels)

        if self.wtih_act:
            self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        if self.with_norm:
            x = self.bn(x)
        if self.wtih_act:
            x = self.act(x)
        return x


def LinearModule(hidden_dim):
    return nn.LayerList(
        [nn.Linear(
            hidden_dim, hidden_dim, bias_attr=True), nn.ReLU()])


class FeatureResize(nn.Layer):
    def __init__(self, size=(10, 25)):
        super(FeatureResize, self).__init__()
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, self.size)
        return x.flatten(2)


class ROIGather(nn.Layer):
    '''
    ROIGather module for gather global information
    Args: 
        in_channels: prior feature channels
        num_priors: prior numbers we predefined
        sample_points: the number of sampled points when we extract feature from line
        fc_hidden_dim: the fc output channel
        refine_layers: the total number of layers to build refine
    '''

    def __init__(self,
                 in_channels,
                 num_priors,
                 sample_points,
                 fc_hidden_dim,
                 refine_layers,
                 mid_channels=48):
        super(ROIGather, self).__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors
        self.f_key = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type='bn')

        self.f_query = nn.Sequential(
            nn.Conv1D(
                in_channels=num_priors,
                out_channels=num_priors,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=num_priors),
            nn.ReLU(), )
        self.f_value = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.W = nn.Conv1D(
            in_channels=num_priors,
            out_channels=num_priors,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=num_priors)

        self.resize = FeatureResize()
        constant_(self.W.weight, 0)
        constant_(self.W.bias, 0)

        self.convs = nn.LayerList()
        self.catconv = nn.LayerList()
        for i in range(refine_layers):
            self.convs.append(
                ConvModule(
                    in_channels,
                    mid_channels, (9, 1),
                    padding=(4, 0),
                    bias=False,
                    norm_type='bn'))

            self.catconv.append(
                ConvModule(
                    mid_channels * (i + 1),
                    in_channels, (9, 1),
                    padding=(4, 0),
                    bias=False,
                    norm_type='bn'))

        self.fc = nn.Linear(
            sample_points * fc_hidden_dim, fc_hidden_dim, bias_attr=True)

        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def roi_fea(self, x, layer_index):
        feats = []
        for i, feature in enumerate(x):
            feat_trans = self.convs[i](feature)
            feats.append(feat_trans)
        cat_feat = paddle.concat(feats, axis=1)
        cat_feat = self.catconv[layer_index](cat_feat)
        return cat_feat

    def forward(self, roi_features, x, layer_index):
        '''
        Args:
            roi_features: prior feature, shape: (Batch * num_priors, prior_feat_channel, sample_point, 1)
            x: feature map
            layer_index: currently on which layer to refine
        Return: 
            roi: prior features with gathered global information, shape: (Batch, num_priors, fc_hidden_dim)
        '''

        roi = self.roi_fea(roi_features, layer_index)
        # return roi
        # print(roi.shape)
        # return roi
        bs = x.shape[0]
        # print(bs)
        #roi = roi.contiguous().view(bs * self.num_priors, -1)
        roi = roi.reshape([bs * self.num_priors, -1])
        # roi = paddle.randn([192,2304])
        # return roi
        # print(roi)
        # print(self.fc)
        # print(self.fc.weight)
        roi = self.fc(roi)
        roi = F.relu(self.fc_norm(roi))
        # return roi
        #roi = roi.view(bs, self.num_priors, -1)
        roi = roi.reshape([bs, self.num_priors, -1])
        query = roi

        value = self.resize(self.f_value(x))  # (B, C, N) global feature
        query = self.f_query(
            query)  # (B, N, 1) sample context feature from prior roi
        key = self.f_key(x)
        value = value.transpose(perm=[0, 2, 1])
        key = self.resize(key)  # (B, C, N) global feature
        sim_map = paddle.matmul(query, key)
        sim_map = (self.in_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, axis=-1)

        context = paddle.matmul(sim_map, value)
        context = self.W(context)

        roi = roi + F.dropout(context, p=0.1, training=self.training)

        return roi


class SegDecoder(nn.Layer):
    '''
    Optionaly seg decoder
    '''

    def __init__(self,
                 image_height,
                 image_width,
                 num_class,
                 prior_feat_channels=64,
                 refine_layers=3):
        super().__init__()
        self.dropout = nn.Dropout2D(0.1)
        self.conv = nn.Conv2D(prior_feat_channels * refine_layers, num_class, 1)
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = F.interpolate(
            x,
            size=[self.image_height, self.image_width],
            mode='bilinear',
            align_corners=False)
        return x


import paddle.nn as nn


def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class)
        target (torch.Tensor): The target of each prediction, shape (N, )
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.shape[0] == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == 2 and target.ndim == 1
    assert pred.shape[0] == target.shape[0]
    assert maxk <= pred.shape[1], \
        f'maxk {maxk} exceeds pred dimension {pred.shape[1]}'
    pred_value, pred_label = pred.topk(maxk, axis=1)
    pred_label = pred_label.t()  # transpose to shape (maxk, N)
    correct = pred_label.equal(target.reshape([1, -1]).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape([-1]).cast("float32").sum(0,
                                                                  keepdim=True)
        correct_k = correct_k * (100.0 / pred.shape[0])
        res.append(correct_k)
    return res[0] if return_single else res


class Accuracy(nn.Layer):
    def __init__(self, topk=(1, ), thresh=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh)
