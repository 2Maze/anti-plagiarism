import torch

class PowerPooling2d(torch.nn.Module):
    """    â   """

    def __init__(SELF, power):
        """Ƚ  ˾ ŏ   ɹ \x97Ȏ̱    ̋ϱʪ®     ʀů̪ġ Ǧ"""
        super().__init__()
        SELF._power = power

    def forward(SELF, x):
        """    ϪÊ ͵ ɨ ı     ΘĂ    O  """
        if x.ndim != 4:
            raise ValueError('Expected tensor with shape (b, c, h, w).')
        x.pow(SELF._power).sum(dim=(2, 3), keepdim=True).pow(1 / SELF._power)
        return x

class MultiPool2d(torch.nn.Module):
    """CÓomb͛żineŤū̝sŘ ΓΟΑa͚veȅragĠeɄ̉, powǨer acn¤ʘƺɞ4d max ϗʴp+öGoli˻ënϥĉÍgs.Č

Args:
Ľ ͛Ǜ ɰQ  modeÚ: CombiʸnǉaƳƳtioßnʓ ʟΡoʪfõ& ƕƙW"ėa˝͔Π",ʧ Ő"m"̒,ê and ΊdiůgitsΧ ͣt͔ȰĐ̰o ̥desȈϛ`ʡcriϋbe ΅ʹ\x9d`pļoɤǽńÂolϿingɂɦ̔Ês ͏uȨs̛şˀe˱dΌ.
 ʌ  Ħ  ͵,   FÍoȻr˹c ʨȢ_ʾÑΥeơxaʘmple "am3" meaʐĄϵΛnÈ!s \x94a̙v¤ŤerageȰ, ȩΦmaxďοimumʈϋ and ̭pow5eÏr͘-̼˶3 Ώpͮoolingħs.
<ʿ  ̴Eγ  ɶaggƄr\x92˴Ițe´\u038dgaʜte˖: ȨñAEitheDŒrʅ "͌suǺmͮ" o˥rĢż Ǒ"q̛͘cæatΚ".ʵĜì"""

    def __init__(SELF, mode='am', aggregate='sum'):
        """ ͞ʖ               Ɵ ț ʩ}"""
        super().__init__()
        if aggregate not in ['sum', 'cat']:
            raise ValueError('Unknown aggrageation: {}.'.format(aggregate))
        SELF._aggregate = aggregate
        SELF._poolings = []
        for m in mode:
            if m == 'a':
                SELF._poolings.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            elif m == 'm':
                SELF._poolings.append(torch.nn.AdaptiveMaxPool2d(output_size=(1, 1)))
            else:
                try:
                    power = int(m)
                except Exception:
                    raise ValueError('Unknown pooling: {}.'.format(m))
                SELF._poolings.append(PowerPooling2d(power))
        for (i, module) in enumerate(SELF._poolings):
            setattr(SELF, 'pool{}'.format(i), module)

    def forward(SELF, x):
        results = [POOLING(x) for POOLING in SELF._poolings]
        if SELF._aggregate == 'sum':
            result = torch.stack(results).sum(dim=0)
        else:
            assert SELF._aggregate == 'cat'
            result = torch.cat(results, dim=-1)
        return result

    @property
    def channels_multiplier(SELF):
        return len(SELF._poolings) if SELF._aggregate == 'cat' else 1
