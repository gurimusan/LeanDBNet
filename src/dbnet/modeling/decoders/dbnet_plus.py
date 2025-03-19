import torch
from torch import nn
import torch.nn.functional as F


class ScaleChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.fc2 = nn.Conv2d(out_planes, num_features, 1, bias=False)
        if init_weight:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = self.avgpool(x)
        global_x = self.fc1(global_x)
        global_x = F.relu(self.bn(global_x))
        global_x = self.fc2(global_x)
        global_x = F.softmax(global_x, 1)
        return global_x


class ScaleChannelSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super().__init__()

        self.channel_wise = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, out_planes , 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_planes, in_planes, 1, bias=False)
        )
        self.spatial_wise = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = self.channel_wise(x).sigmoid()
        global_x = global_x + x
        x = torch.mean(global_x, dim=1, keepdim=True)
        global_x = self.spatial_wise(x) + global_x
        global_x = self.attention_wise(global_x)
        return global_x


class ScaleSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super().__init__()

        self.spatial_wise = nn.Sequential(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = torch.mean(x, dim=1, keepdim=True)
        global_x = self.spatial_wise(global_x) + x
        global_x = self.attention_wise(global_x)
        return global_x


class ScaleFeatureSelection(nn.Module):
    def __init__(self, in_channels, inter_channels , out_features_num=4, attention_type='scale_spatial'):
        super().__init__()

        self.in_channels=in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.type = attention_type
        if self.type == 'scale_spatial':
            self.enhanced_attention = ScaleSpatialAttention(inter_channels, inter_channels//4, out_features_num)
        elif self.type == 'scale_channel_spatial':
            self.enhanced_attention = ScaleChannelSpatialAttention(inter_channels, inter_channels // 4, out_features_num)
        elif self.type == 'scale_channel':
            self.enhanced_attention = ScaleChannelAttention(inter_channels, inter_channels//2, out_features_num)

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, concat_x, features_list):
        concat_x = self.conv(concat_x)
        score = self.enhanced_attention(concat_x)
        assert len(features_list) == self.out_features_num
        if self.type not in ['scale_channel_spatial', 'scale_spatial']:
            shape = features_list[0].shape[2:]
            score = F.interpolate(score, size=shape, mode='bilinear')
        x = []
        for i in range(self.out_features_num):
            x.append(score[:, i:i+1] * features_list[i])
        return torch.cat(x, dim=1)


class DBNetPlusDecoder(nn.Module):
    def __init__(
        self,
        in_channels: list[int]=[64, 128, 256, 512],
        inner_channels: int=256,
        k: int=50,
        smooth: bool=False,
        serial: bool=False,
        attention_type: str = "scale_spatial",
        ):
        """
        :param int k: Controll the gradient.
        :param bool smooth: If true, use bilinear instead of deconv.
        :param bool serial: If true, thresh prediction will combine segmentation result as input.
        """
        super().__init__()

        self.k = k
        self.serial = serial

        # for feature pyramid network
        self.in5 = nn.Conv2d(in_channels[3], inner_channels, kernel_size=1, bias=False)
        self.in4 = nn.Conv2d(in_channels[2], inner_channels, kernel_size=1, bias=False)
        self.in3 = nn.Conv2d(in_channels[1], inner_channels, kernel_size=1, bias=False)
        self.in2 = nn.Conv2d(in_channels[0], inner_channels, kernel_size=1, bias=False)
        self.up5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels//4, kernel_size=3, padding=1, bias=False)

        self.concat_attention = ScaleFeatureSelection(
            inner_channels, inner_channels//4, attention_type=attention_type)

        # for probability map
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())

        self.binarize.apply(self._init_weights)

        # for threshold map
        self.thresh = self._init_thresh(
            inner_channels, serial=serial, smooth=smooth)
        self.thresh.apply(self._init_weights)

        self.in5.apply(self._init_weights)
        self.in4.apply(self._init_weights)
        self.in3.apply(self._init_weights)
        self.in2.apply(self._init_weights)
        self.out5.apply(self._init_weights)
        self.out4.apply(self._init_weights)
        self.out3.apply(self._init_weights)
        self.out2.apply(self._init_weights)

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels: int, serial: bool=False, smooth: bool=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        return nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth),
            nn.Sigmoid())

    def _init_upsample(self, in_channels, out_channels, smooth=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_channels, inter_out_channels, kernel_size=3, stride=1, padding=1, bias=False)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=0, bias=False))

            return nn.Sequential(*module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def _differentiable_binarization(self, p, t):
        """`B = 1 / (1 + exp(-k * (p - t)))`

        Args:
            p: probability map
            t: threshold map
        """
        return torch.reciprocal(1 + torch.exp(-self.k * (p - t)))

    def forward(self, x: torch.Tensor):
        c2, c3, c4, c5 = x # x2, x3, x4, x5 = 1/4, 1/8, 1/16, 1/32

        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = in4 + self.up5(in5)  # 1/16
        out3 = in3 + self.up4(out4) # 1/8
        out2 = in2 + self.up3(out3) # 1/4

        p5 = self.out5(in5)     # 1/4
        p4 = self.out4(out4)    # 1/4
        p3 = self.out3(out3)    # 1/4
        p2 = self.out2(out2)    # 1/4

        fuse = torch.cat((p5, p4, p3, p2), 1)
        fuse = self.concat_attention(fuse, [p5, p4, p3, p2])

        result: dict[str, any] = dict()

        # pred text segmentation map
        prob_map = self.binarize(fuse)
        result.update(prob_map=prob_map)

        # in inference, return only the text segmentation map
        if not self.training:
            return result

        # in training,
        if self.serial:
            fuse = torch.cat(
                    (fuse, nn.functional.interpolate(
                        prob_map, fuse.shape[2:])), 1)

        # pred threshold map
        thresh_map = self.thresh(fuse)
        result.update(thresh_map=thresh_map)

        # calc a approximate binary map
        binary_map = self._differentiable_binarization(prob_map, thresh_map)
        result.update(binary_map=binary_map)

        return result
