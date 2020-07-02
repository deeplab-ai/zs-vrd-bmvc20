"""Import all different models here."""

import torch

from .sg_generator import (
    ATRNet, LangSpatNet, LanguageNet, RelDN, SpatialNet, UVTransE, VisualNet,
    VisualSpatAttentionNet, VisualSpatNet
)
from .sg_projector import (
    LangSpatProjector, LanguageProjector, SpatialProjector, VisualProjector,
    VisualSpatAttentionProjector, VisualSpatProjector
)

MODELS = {
    'atr_net': ATRNet,
    'language_net': LanguageNet,
    'language_projector': LanguageProjector,
    'lang_spat_net': LangSpatNet,
    'lang_spat_projector': LangSpatProjector,
    'reldn_net': RelDN,
    'spatial_net': SpatialNet,
    'spatial_projector': SpatialProjector,
    'uvtranse_net': UVTransE,
    'visual_net': VisualNet,
    'visual_projector': VisualProjector,
    'visual_spat_attention_net': VisualSpatAttentionNet,
    'visual_spat_attention_projector': VisualSpatAttentionProjector,
    'visual_spat_net': VisualSpatNet,
    'visual_spat_projector': VisualSpatProjector
}


def load_model(config, model, net_name, path=None):
    """Load model given a net name."""
    model_dir = config.paths['models_path']\
        if path is None else path
    net = MODELS[model](config)
    checkpoint = torch.load(model_dir + net_name + '.pt',
                            map_location=config.device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return net.to(config.device)
