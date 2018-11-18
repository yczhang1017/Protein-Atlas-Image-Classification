
import os
import numpy as np
import pandas as pd
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
#from torchvision.transforms import transforms
#from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Protain Alta Image Classification')
parser.add_argument('--root', default='./',
                    type=str, help='directory of the data')
parser.add_argument('--batch_size', default=24, type=int,
                    help='Batch size for training')
parser.add_argument('--workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of epochs to train')
parser.add_argument('--save_folder', default='save/', type=str,
                    help='Dir to save results')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay')
parser.add_argument('--step_size', default=10, type=int,
                    help='Number of steps for every learning rate decay')
parser.add_argument('--checkpoint', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='epoch number to be resumed at')
parser.add_argument('--type', default='A',  choices=['A', 'B'], type=str,
                    help='type of the model')


args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
    
NAME = {
    0:  "Nucleoplasm", 
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center" ,  
    4:  "Nuclear speckles"   ,
    5:  "Nuclear bodies"   ,
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus"   ,
    8:  "Peroxisomes"   ,
    9:  "Endosomes"   ,
    10:  "Lysosomes"   ,
    11:  "Intermediate filaments",   
    12:  "Actin filaments"   ,
    13:  "Focal adhesion sites",   
    14:  "Microtubules"   ,
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle"   ,
    18:  "Microtubule organizing center" ,  
    19:  "Centrosome"   ,
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions"  , 
    23:  "Mitochondria"   ,
    24:  "Aggresome"   ,
    25:  "Cytosol",
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings" 
    }
NLABEL=len(NAME)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device("cpu")

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self,root,phase,image_labels=None, size=None ,transform=None):
        self.root=os.path.expanduser(root)
        self.phase=phase
        self.transform = transform
        self.size=size
        if not phase=='test':
            self.img_dir=os.path.join(root,'train_img')
        else:
            self.img_dir=os.path.join(root,'test_img')
        self.colors=['red','green','blue','yellow']
        self.image_labels=image_labels
        if image_labels==None:
            files=os.listdir(self.img_dir)
            for img in files:
                if img.endswith('blue.png'):
                    ims=img.split('_')
                    self.images+=[ims[0]]
        #csv_file=os.path.join(root,'train.csv')
        #self.labels=pd.read_csv(csv_file, index_col=0, squeeze=True).to_dict()
        #for key in self.labels.keys():
        #    self.labels[key]=[int(a) for a in self.labels[key].split(' ')]

    def __getitem__(self, index):
        if not self.phase in ['test']:
            img_id,labels=self.image_labels[index]
        else:
            img_id=self.image_labels[index]
            
        im_tensor=torch.zeros((len(self.colors),512,512),device="cpu")
        for j,color in enumerate(self.colors):
            image_dir=os.path.join(self.img_dir,img_id+'_'+color+'.png')
            image=imageio.imread(image_dir)
            im_tensor[j,:,:]=torch.tensor(image,dtype=torch.float,device="cpu")/256
        if self.size is not None:
            im_tensor=F.adaptive_avg_pool2d(im_tensor, self.size)    
        if self.transform is not None:
            im_tensor = self.transform(im_tensor)
        
        if not self.phase in ['test']:
            target=torch.zeros((NLABEL),device="cpu")
            for label in labels:
                target[label]=1
            return (im_tensor,target)
        else:
            return im_tensor
        
    def __len__(self):
        return len(self.image_labels)

 
'''
custom-built SqueezeNet model
'''
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls,Bottleneck,BasicBlock
import collections
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return self.sigmoid(x)
                
'''
Focal loss to handle imbalance between foreground and background
from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
https://arxiv.org/pdf/1708.02002.pdf
'''
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets,reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets,reduction='none')
        pt = torch.exp(-BCE_loss).detach()
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
    
    
def main():
    csv_file=os.path.join(args.root,'train.csv')
    label_dict=pd.read_csv(csv_file, index_col=0, squeeze=True).to_dict()
    for key,label in label_dict.items():
        label_dict[key]=[int(a) for a in label.split(' ')]
        

    ids={i:[] for i in range(NLABEL)}
    for key,label in label_dict.items():
        for j in label:
            ids[j].append(key)
    #repeat training images with rare labels
    repeat=[];
    for i in range(NLABEL):
        repeat.append(int(np.power(len(label_dict)/len(ids[i]),0.2)))
    repeat=np.array(repeat)
    #Divide image ids into training and evaluation parts
    image_sets={'train': set(label_dict.keys()),
                 'val':set([]) }
    for l,ims in ids.items():
        ll=len(ims)
        vl=int(np.ceil(ll*0.05))
        varray=np.random.choice(ll, vl, replace=False)
        for i in varray:
            im=ims[i]
            image_sets['val'].add(im)
            image_sets['train'].discard(im)
    
    image_labels={'train':[], 'val':[]}
    for phase in ['train','val']:
        for im in image_sets[phase]:
            im_label=label_dict[im]
            
            label_array=np.array(im_label)
            im_repeat= np.max(repeat[label_array])
            if phase=='val':
                image_labels[phase].append((im,im_label))
            else:
                for i in range(im_repeat):
                    image_labels[phase].append((im,im_label))
                    

    dataset={x: ProteinDataset(args.root,x,image_labels[x]) 
            for x in ['train', 'val']}
    dataloader={x: torch.utils.data.DataLoader(dataset[x],
            batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
            for x in ['train', 'val']}
    #dataset_sizes={x: len(dataset[x]) for x in ['train', 'val']}
    #model = VGG(make_layers(cfg[args.type], batch_norm=True))
    #model =SqueezeNet(version=1.1)
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    
    pre_trained=model_zoo.load_url(model_urls['resnet34'])
    con1_weight=pre_trained['conv1.weight']
    pre_trained['conv1.weight']=torch.cat((con1_weight,con1_weight[:,1,:,:].view(64,1,7,7)),1)
    pre_trained['fc.weight']=pre_trained['fc.weight'][:NLABEL,:]
    pre_trained['fc.bias']=pre_trained['fc.bias'][:NLABEL]
    '''
    pre_trained2=collections.OrderedDict()
    for _ in range(len(pre_trained)):
        k,v=pre_trained.popitem(last=False)
        if k.startswith("features."):
            k=k.replace("features.","")
            k=k.replace("12","13")
            k=k.replace("11","12")
            pre_trained2[k]=v
    '''     
    model.load_state_dict(pre_trained)
    
    if torch.cuda.is_available():
        model=nn.DataParallel(model)
        cudnn.benchmark = True
        
    if args.checkpoint:
        print('Resuming training, loading {}...'.format(args.checkpoint))
        weight_file=os.path.join(args.root,args.checkpoint)
        model.load_state_dict(torch.load(weight_file,
                                 map_location=lambda storage, loc: storage))
        
    if torch.cuda.is_available():
        model = model.cuda()
    
    criterion = FocalLoss()
    optimizer = optim.SGD(model.parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    t00 = time.time()
    best_F1=0.0
    
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 5)
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            
            running_loss=0
            running_F1=0
            num=0 
            for inputs,targets in dataloader[phase]:
                t01 = time.time()
                inputs = inputs.to(device)                
                targets= targets.to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                num += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                propose=(outputs>0.5)
                targets=targets.byte()
                corrects= torch.sum(propose==targets,1).double()
                selected= torch.sum(propose,1).double()
                relevant= torch.sum(targets,1).double()
                if torch.sum(corrects==0)==0:
                    F1=2/(selected/corrects+relevant/corrects)
                    running_F1 +=torch.sum(F1).item()
                else:
                    corrects=corrects.cpu().numpy()
                    s=selected.cpu().numpy()
                    r=relevant.cpu().numpy()
                    for i,c in enumerate(corrects):
                        if c>0:
                            running_F1 += 2/(s[i]/c+r[i]/c)
                if phase=='eval':
                    import pdb; pdb.set_trace()
                average_loss = running_loss/num
                average_F1 = running_F1/num
                t02 = time.time()
                if num % (100*inputs.size(0))==0:
                    print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'.format(
                            num, average_loss, average_F1, t02-t01))
            if phase == 'val' and average_F1 > best_F1:
                best_F1 = average_F1
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),os.path.join(args.root,args.save_folder,'out_'+str(epoch+1)+'.pth'))
        print()
        
if __name__ == '__main__':
    main()        


