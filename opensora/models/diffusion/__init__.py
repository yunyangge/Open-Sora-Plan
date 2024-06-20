
from .latte.modeling_latte import Latte_models
from .latte.modeling_inpaint import inpaint_models

Diffusion_models = {}
Diffusion_models.update(Latte_models)
Diffusion_models.update(inpaint_models)

    