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
        skip_downsample,
        image_height,
        image_width,
        model_name: str = 'resnet50', 
        return_interm_layers: bool = True, 
        return_stem: bool = False,
        dilation: bool = False, 
        pretrained: bool = True,
        freeze: bool = False,
        return_stages: int = 5
    ):
        super().__init__()
        assert model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        self.model_name =  model_name
        backbone = getattr(torchvision.models, model_name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained)

        if use_our_ckpt:
            print('\n=================================================\n USING OUR CHECKPOINTS\n=================================================\n')
            ck = torch.load(ckpt_path, map_location=torch.device('cpu'))
            output_dict = dict(state_dict=dict())
            for key, value in ck['state_dict'].items():
                if key.startswith('backbone'):
                    output_dict['state_dict'][key[9:]] = value
            
            backbone.load_state_dict(output_dict['state_dict'], strict=False)


        if skip_downsample:

            backbone.maxpool = nn.Identity()

            # backbone.layer2[0].conv2.stride = (1,1)
            # backbone.layer2[0].downsample[0].stride = (1,1)
            print('\n=================================================\n SKIP DOWNSAMPLE\n=================================================\n')


        # print(backbone)
    
        # for param_tensor in backbone.state_dict():
        #     if 'layer1.1' in param_tensor:
        #         print(param_tensor, "\t", backbone.state_dict()[param_tensor])
        

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
        return [output[3], output[4]]




if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224).cuda()
    model = ResNetExtractor(
        model_name='resnet50', 
        return_interm_layers=True, 
        dilation=False,
        pretrained=True,
        use_our_ckpt=False,
        skip_layer2_downsample=True,
        ckpt_path='/home/jerryyuan/MyCVT/cross_view_transformer/checkpoints/densecl_aco_r50_epoch_100.pth',
        image_height=224,
        image_width=224)
    # print(model)
    model = model.cuda()

    print(model(x)[1].shape)

    # y = model(x)
    # for k, v in y.items():
    #     print(k, v.shape)    
    # model2 = ResNetExtractor()
    # model2 = model2.cuda()
    # y2 = model2(x)