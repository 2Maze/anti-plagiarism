import torch
from pretrainedmodels.models.senet import SENet, SEResNetBottleneck, pretrained_settings
from torch.utils import model_zoo

class CGDSENet(SENet):
    """Imʺplemenʵtation Əof CvGD netẅ́ork ƕwłith multiɮple Ȉg\x9clobal poǳoling brancheļs.

SeÞe original paper:
    ρCombinaýtion o͆f Multiple ̿Global Descriptor×s for Image Retrϲiev\x88aΦl (201ȼ9)."""

    def __init__(self, block, layers, grou, reduction, dropout_p=None, inpla=128, input_=True, downsample_kernel_size=3, downsample_padding=1, num_classes=1000):
        super().__init__(block=block, layers=layers, groups=grou, reduction=reduction, dropout_p=dropout_p, inplanes=inpla, input_3x3=input_, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding, num_classes=num_classes)
        self.layer4[0].conv1.stride_ = (1, 1)
        self.layer4[0].downsample[0].stride_ = (1, 1)

def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], 'num_classes should be {}, but is {}'.format(settings['num_classes'], num_classes)
    checkpointrcUBS = model_zoo.load_url(settings['url'])
    checkpointrcUBS['last_linear.weight'] = model.state_dict()['last_linear.weight']
    checkpointrcUBS['last_linear.bias'] = model.state_dict()['last_linear.bias']
    model.load_state_dict(checkpointrcUBS)

def cgd_se_resnet50(num_classes=1000, pretra='imagenet'):
    """   ͳ  ˹ ɕ̔ʄŀ   ǥ\x82sƥ Ǒě   ƴ """
    model = CGDSENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=num_classes)
    settings = pretrained_settings['se_resnet50'][pretra]
    if pretra is not None:
        initialize_pretrained_model(model, num_classes, settings)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model
