# wisdom/utils/patched.py
import os
import torch
import torchvision.models as models
from .common import get_model

# Quick patch for torchvision resnet models
# ResNet implmented in torchvision is not suitable for the Captum attribution methods, a little customizations are made in the model definition.
def patch_resnet_torchvision(src, dst_model):
    src_sd = src.state_dict() if isinstance(src, torch.nn.Module) else src
    dst_sd = dst_model.state_dict()

    def rename(key: str) -> str:
        return (key
                .replace(".downsample.0.", ".shortcut_conv.")
                .replace(".downsample.1.", ".shortcut_bn."))

    patched = {rename(k): v for k, v in src_sd.items()
               if rename(k) in dst_sd and v.shape == dst_sd[rename(k)].shape}

    merged = dst_sd.copy()
    merged.update(patched)
    return merged

def patch_resnet_imagenet(dst_model, saved_model_weight='./models_info/saved_models/resnet18_IMAGENET_whole.pth', model_name='resnet18'):
    
    if os.path.exists(saved_model_weight):
         src_model, _, _ = get_model(saved_model_weight)
    else:
        print(f"Model weight file {saved_model_weight} does not exist, try to download it first.")
        model_func = getattr(models, model_name)
        src_model = model_func(weights="IMAGENET1K_V1") 
    
    src_model.eval()
    dst_model.eval()
    
    patched_sd  = patch_resnet_torchvision(src_model, dst_model)
    incompat = dst_model.load_state_dict(patched_sd, strict=False)
    assert incompat.missing_keys == [], f"Missing keys in destination model: {incompat.missing_keys}"
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        d = (src_model(x) - dst_model(x)).abs().max().item()
        assert d < 1e-5, f"Model patching failed, max diff: {d}"
    
    torch.save(dst_model.state_dict(), './models_info/saved_models/{}_IMAGENET_patched.pt'.format(model_name))
    torch.save(dst_model, './models_info/saved_models/{}_IMAGENET_patched_whole.pth'.format(model_name))
    
    return dst_model 