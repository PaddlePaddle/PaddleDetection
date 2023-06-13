from typing import Dict
from collections import OrderedDict
from ppdet.modeling.architectures.meta_arch import BaseArch


class MultiSteamDetector(BaseArch):
    def __init__(self,
                 model: Dict[str, BaseArch],
                 train_cfg=None,
                 test_cfg=None):
        super(MultiSteamDetector, self).__init__()
        self.submodules = list(model.keys())
        for k, v in model.items():
            setattr(self, k, v)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.inference_on = self.test_cfg.get("inference_on",
                                              self.submodules[0])
        self.first_load = True

    def forward(self, inputs, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(inputs, **kwargs)
        else:
            return self.forward_test(inputs, **kwargs)

    def get_loss(self, **kwargs):
        # losses = self(**data)

        return self.forward_train(self, **kwargs)

    def model(self, **kwargs) -> BaseArch:
        if "submodule" in kwargs:
            assert (kwargs["submodule"] in self.submodules
                    ), "Detector does not contain submodule {}".format(kwargs[
                        "submodule"])
            model: BaseArch = getattr(self, kwargs["submodule"])
        else:
            model: BaseArch = getattr(self, self.inference_on)
        return model

    def freeze(self, model_ref: str):
        assert model_ref in self.submodules
        model = getattr(self, model_ref)
        model.eval()
        for param in model.parameters():
            param.stop_gradient = True

    def update_ema_model(self, momentum=0.9996):
        # print(momentum)
        model_dict = self.student.state_dict()
        new_dict = OrderedDict()
        for key, value in self.teacher.state_dict().items():
            if key in model_dict.keys():
                new_dict[key] = (model_dict[key] *
                                 (1 - momentum) + value * momentum)
            else:
                raise Exception("{} is not found in student model".format(key))
        self.teacher.set_dict(new_dict)
