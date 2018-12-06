import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.vgg import make_layers
NLABEL=28
'''
Resnet
'''

class VGG(nn.Module):
    def __init__(self, features, num_classes=NLABEL, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
                
                
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     groups=groups, bias=False)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=NLABEL):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        #self.dropout2d = nn.Dropout2d()
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.sigmoid=nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool(x) #added maxpool
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.dropout2d(x) #added dropout
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) #added dropout
        x = self.fc(x)

        return x
'''
Inception 
'''
from torchvision.models.inception import BasicConv2d,InceptionA,InceptionB,InceptionC,InceptionD,InceptionE,InceptionAux

class Inception3(nn.Module):
    def __init__(self, num_classes=NLABEL, aux_logits=False, transform_input=False):
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(4, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # 512 x 512 x 4
        x = self.Conv2d_1a_3x3(x)
        # 255 x 255 x 32
        x = self.Conv2d_2a_3x3(x)
        # 253 x 253 x 32
        x = self.Conv2d_2b_3x3(x)
        # 253 x 253 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 121 x 121 x 64
        x = self.Conv2d_3b_1x1(x)
        # 121 x 121 x 80
        x = self.Conv2d_4a_3x3(x)
        # 119 x 119 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 59 x 59 x 192
        x = self.Mixed_5b(x)
        # 59 x 59 x 256
        x = F.max_pool2d(x, kernel_size=3, stride=2) #added maxpool
        # 29 x 29 x 256
        x = self.Mixed_5c(x)
        # 29 x 29 x 256
        x = self.Mixed_5d(x)
        # 29 x 29 x 256
        x = self.Mixed_6a(x)
        # 14 x 14 x 768
        x = self.Mixed_6b(x)
        # 14 x 14 x 768
        x = self.Mixed_6c(x)
        # 14 x 14 x 768
        x = self.Mixed_6d(x)
        # 14 x 14 x 768
        x = self.Mixed_6e(x)
        # 14 x 14 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 14 x 14 x 768
        x = self.Mixed_7a(x)
        # 7 x 7 x 1280
        x = self.Mixed_7b(x)
        # 7 x 7 x 2048
        x = self.Mixed_7c(x)
        # 7 x 7 x 2048
        x = self.avgpool(x)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x

'''
SENet
'''
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
    
class XBottleneck(nn.Module):
    expansion = 2
    reduction=16
    def __init__(self,inplanes, planes, stride=1,groups=32, SE=True, downsample=None):
        super(XBottleneck, self).__init__()
        planes2= planes * self.expansion
        self.conv1 = conv1x1(inplanes, planes,groups=groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride,groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes2 ,groups=groups)
        self.bn3 = nn.BatchNorm2d(planes2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.SE = SE
        self.SELayer = SELayer(planes2, self.reduction)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.SE:
            out = self.SELayer(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SENet(nn.Module):
    def __init__(self, block, layers, num_classes=NLABEL):
        super(SENet, self).__init__()
        self.inplanes = 64
        self.bloc1 = nn.Sequential(
                conv3x3(4, self.inplanes, stride=2),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
                conv3x3(self.inplanes, self.inplanes, stride=1),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
                conv3x3(self.inplanes, 2*self.inplanes, stride=2),
                nn.BatchNorm2d(2*self.inplanes),
                nn.ReLU(inplace=True)
        )
        self.inplanes= 128
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(1024*block.expansion, num_classes)
        self.sigmoid=nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3(self.inplanes, planes * block.expansion, stride, groups=32), #SENet modification
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bloc1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) #added dropout
        x = self.fc(x)

        return x

from torchvision.models.vgg import model_urls as vgg_url
from torchvision.models.resnet import model_urls as resnet_uls
from torchvision.models.resnet import BasicBlock,Bottleneck
from torchvision.models.inception import model_urls as inception_url

def CNN_models(model_type):
    model_url=None; con1_name=None
    if model_type=='res34':
        model = ResNet(BasicBlock, [3, 4, 6, 3])
        model_url=resnet_uls['resnet34']
        con1_name='conv1.weight'
    elif model_type=='res50':
        model = ResNet(Bottleneck, [3, 4, 6, 3])
        model_url=resnet_uls['resnet50']
        con1_name='conv1.weight'
    elif model_type=='senet':
        model = SENet(XBottleneck, [3, 4, 6, 3])
    elif model_type=='inception':
        model = Inception3()
        model_url=inception_url['inception_v3_google']
        con1_name='Conv2d_1a_3x3.conv.weight'
    elif model_type=='vgg16bn':
        model =VGG(make_layers([64, 64, 'M', 128, 128, 'M',
                256, 256, 256, 'M', 512, 512, 'M', 
                512, 512,'M',512, 512, 'M'], batch_norm=False))
        model_url=vgg_url['vgg16']
        con1_name='features.0.weight'
    
    return (model,model_url,con1_name)

