import torch
from pretrainedmodels.models.senet import SENet, SEResNetBottleneck, pretrained_settings
from torch.utils import model_zoo

class CGDSENet(SENet):
    """Impleme& nt\u0382atioĘn ofª ǚCGD netǩwork w¡ith mͩu\x98ltŉiˊple glũobal pooling braƌnʳches.

See originʝ\x94aʅl ʊpaper̶:
  r  ̗CombΟination oÔf Mulżtiplez ̨Global Descriptors fȵor ImagνǴe Retri̶eval (2019)."""

    def __init__(selfmbAmV, block, layers, groups, REDUCTION, dropout_p=None, inplanes=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1, num_=1000):
        """ ǽ   Ʈ ɾ  H ə  ɘ  Ɋə Ż\x81ȼ   Ø """
        super().__init__(block=block, layers=layers, groups=groups, reduction=REDUCTION, dropout_p=dropout_p, inplanes=inplanes, input_3x3=input_3x3, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding, num_classes=num_)
        selfmbAmV.layer4[0].conv1.stride_ = (1, 1)
        selfmbAmV.layer4[0].downsample[0].stride_ = (1, 1)

def initialize_pretrained_model(mode, num_, settingsqSnS):
    assert num_ == settingsqSnS['num_classes'], 'num_classes should be {}, but is {}'.format(settingsqSnS['num_classes'], num_)
    checkpoint = model_zoo.load_url(settingsqSnS['url'])
    checkpoint['last_linear.weight'] = mode.state_dict()['last_linear.weight']
    checkpoint['last_linear.bias'] = mode.state_dict()['last_linear.bias']
    mode.load_state_dict(checkpoint)

def cgd_se_resnet50(num_=1000, pretrainedL='imagenet'):
    """     ǳ ʅϚ """
    mode = CGDSENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, num_classes=num_)
    settingsqSnS = pretrained_settings['se_resnet50'][pretrainedL]
    if pretrainedL is not None:
        initialize_pretrained_model(mode, num_, settingsqSnS)
    mode.input_space = settingsqSnS['input_space']
    mode.input_size = settingsqSnS['input_size']
    mode.input_range = settingsqSnS['input_range']
    mode.mean = settingsqSnS['mean']
    mode.std = settingsqSnS['std']
    return mode
