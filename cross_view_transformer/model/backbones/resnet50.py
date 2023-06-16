import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from collections import OrderedDict
from resnet_pytorch import ResNet

# Precomputed aliases
MODELS = {
    'resnet50'
}

class ResNetExtractor(nn.Module):
    def __init__(
        self, 
        use_our_ckpt,
        ckpt_path,
        image_height,
        image_width,
        model_name: str = 'resnet50', 
        return_interm_layers: bool = True, 
        return_stem: bool = False,
        dilation: bool = False, 
        pretrained: bool = True,
        freeze: bool = False,
        return_stages: int = 3
    ):
        super().__init__()
        assert model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        self.model_name =  model_name
        backbone = getattr(torchvision.models, model_name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained)

        if use_our_ckpt:
            checkpoint = torch.load(ckpt_path)
            backbone.load_state_dict(checkpoint['state_dict'], strict=False)
        

        self.return_interm_layers = return_interm_layers
        self.return_stem = return_stem
        if return_interm_layers:
            return_layers = OrderedDict()
            start_idx = 0
            if return_stem:
                # max_pool is the output of stem layer
                return_layers = {'maxpool': 0}
                return_stages += 1
                start_idx += 1
                print("return_stem is True, return_stages is set to {}".format(return_stages))
            for i in range(1, return_stages):
                return_layers['layer{}'.format(i)] = i
        else:
            return_layers = {'layer4': 0}

        backbone.fc = nn.Identity()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.return_stages = return_stages
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        dummy = torch.rand(1, 3, image_height, image_width)
        output_shapes = [x.shape for x in self(dummy)]

        self.output_shapes = output_shapes

    @property
    def latent_dim(self):
        dims = {
            'resnet18': [64, 128, 256, 512],
            'resnet34': [64, 128, 256, 512],
            'resnet50': [256, 512, 1024, 2048],
            'resnet101': [256, 512, 1024, 2048],
            'resnet152': [256, 512, 1024, 2048]
        }[self.model_name]
        if self.return_stem:
            dims = [64] + dims
        return sum(dims[:self.return_stages])

    def forward(self, x):
        output = self.body(x)
        # we only need the second and third stage for a 8x and 16x down scaling
        return [output[1], output[2]]




if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224).cuda()
    model = ResNetExtractor(
        model_name='resnet50', 
        return_interm_layers=True, 
        dilation=False,
        pretrained=False)
    print(model)
    model = model.cuda()
    y = model(x)
    for k, v in y.items():
        print(k, v.shape)    
    model2 = ResNetExtractor()
    model2 = model2.cuda()
    y2 = model2(x)