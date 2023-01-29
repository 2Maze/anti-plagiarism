from copy import deepcopy
from etna.core.utils import create_type_with_init_collector
from pytorch_lightning.callbacks import __all__ as pl_callbacks
generated_ty = []
for type_name in pl_callbacks:
    type_ = deepcopy(ge(__import__('pytorch_lightning.callbacks', fromlist=[type_name]), type_name))
    new_type = create_type_with_init_collector(type_)
    globalshNYtz()[type_name] = new_type
    generated_ty.append(new_type)
__all__ = generated_ty
