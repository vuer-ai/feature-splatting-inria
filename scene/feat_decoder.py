import torch
import torch.nn as nn
import torch.nn.functional as F

class feat_decoder(nn.Module):
    def __init__(self, feature_shape_list):
        super(feat_decoder, self).__init__()
        assert len(feature_shape_list) >= 2
        for ele in feature_shape_list:
            assert isinstance(ele, int)
        
        self.mod_list = []

        for i in range(len(feature_shape_list) - 1):
            input_dim = feature_shape_list[i]
            output_dim = feature_shape_list[i + 1]
            self.mod_list.append(nn.utils.weight_norm(nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0)))
            # self.mod_list.append(nn.BatchNorm2d(output_dim))
            if i != len(feature_shape_list) - 2:
                self.mod_list.append(nn.ReLU(inplace=True))
        
        # Initialize conv layers as identity
        for i in range(0, len(self.mod_list), 2):
            nn.init.normal_(self.mod_list[i].weight)
            nn.init.normal_(self.mod_list[i].bias)
        
        self.net = nn.Sequential(*self.mod_list)

    def forward(self, x):
        x = self.net(x)
        return x

class skip_feat_decoder(nn.Module):
    def __init__(self, input_dim, part_level=False):
        super(skip_feat_decoder, self).__init__()
        self.part_level = part_level
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(input_dim, 256, kernel_size=1, stride=1, padding=0))
        self.dino_conv = nn.utils.weight_norm(nn.Conv2d(256, 384, kernel_size=1, stride=1, padding=0))
        self.clip_conv1 = nn.utils.weight_norm(nn.Conv2d(256, 768, kernel_size=1, stride=1, padding=0))
        
        # Initialize conv layers as identity
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv1.bias)
        nn.init.normal_(self.dino_conv.weight)
        nn.init.normal_(self.dino_conv.bias)
        nn.init.normal_(self.clip_conv1.weight)
        nn.init.normal_(self.clip_conv1.bias)

        if self.part_level:
            self.part_conv1 = nn.utils.weight_norm(nn.Conv2d(256, 768, kernel_size=1, stride=1, padding=0))
            nn.init.normal_(self.part_conv1.weight)
            nn.init.normal_(self.part_conv1.bias)

    def forward(self, x):
        intermediate_feature = self.conv1(x)
        intermediate_feature = F.relu(intermediate_feature)
        dino_feature = self.dino_conv(intermediate_feature)
        clip_feature = self.clip_conv1(intermediate_feature)
        if self.part_level:
            part_feature = self.part_conv1(intermediate_feature)
            return dino_feature, clip_feature, part_feature
        else:
            return dino_feature, clip_feature
