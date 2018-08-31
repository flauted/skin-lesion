# The torchvision initializers suck because
# they don't allow user to specify a path.
# This is a hot fix

import torchvision.models.vgg as vgg
import torch.utils.model_zoo as model_zoo


# modified from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg19_bn(pretrained=False, model_dir=None, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        model_dir (None, str): If not None, specifies the directory for the model pickle.
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = vgg.VGG(vgg.make_layers(vgg.cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(vgg.model_urls['vgg19_bn'], model_dir=model_dir))
    return model
