class NameAdapter(object):
    """Fix the backbones variable names for pretrained weight"""

    def __init__(self, model):
        super(NameAdapter, self).__init__()
        self.model = model

    @property
    def model_type(self):
        return getattr(self.model, '_model_type', '')

    @property
    def variant(self):
        return getattr(self.model, 'variant', '')

    def fix_conv_norm_name(self, name):
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        # the naming rule is same as pretrained weight
        if self.model_type == 'SEResNeXt':
            bn_name = name + "_bn"
        return bn_name

    def fix_shortcut_name(self, name):
        if self.model_type == 'SEResNeXt':
            name = 'conv' + name + '_prj'
        return name

    def fix_bottleneck_name(self, name):
        if self.model_type == 'SEResNeXt':
            conv_name1 = 'conv' + name + '_x1'
            conv_name2 = 'conv' + name + '_x2'
            conv_name3 = 'conv' + name + '_x3'
            shortcut_name = name
        else:
            conv_name1 = name + "_branch2a"
            conv_name2 = name + "_branch2b"
            conv_name3 = name + "_branch2c"
            shortcut_name = name + "_branch1"
        return conv_name1, conv_name2, conv_name3, shortcut_name

    def fix_basicblock_name(self, name):
        if self.model_type == 'SEResNeXt':
            conv_name1 = 'conv' + name + '_x1'
            conv_name2 = 'conv' + name + '_x2'
            shortcut_name = name
        else:
            conv_name1 = name + "_branch2a"
            conv_name2 = name + "_branch2b"
            shortcut_name = name + "_branch1"
        return conv_name1, conv_name2, shortcut_name

    def fix_layer_warp_name(self, stage_num, count, i):
        name = 'res' + str(stage_num)
        if count > 10 and stage_num == 4:
            if i == 0:
                conv_name = name + "a"
            else:
                conv_name = name + "b" + str(i)
        else:
            conv_name = name + chr(ord("a") + i)
        if self.model_type == 'SEResNeXt':
            conv_name = str(stage_num + 2) + '_' + str(i + 1)
        return conv_name

    def fix_c1_stage_name(self):
        return "res_conv1" if self.model_type == 'ResNeXt' else "conv1"
