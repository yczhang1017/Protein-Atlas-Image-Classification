
import os
import numpy as np
import pandas as pd
#import imageio
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
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
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=2e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of epochs to train')
parser.add_argument('--save_folder', default='save/', type=str,
                    help='Dir to save results')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay')
parser.add_argument('--step_size', default=4, type=int,
                    help='Number of steps for every learning rate decay')
parser.add_argument('--checkpoint', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='epoch number to be resumed at')
parser.add_argument('--model', default='resnet',  choices=['resnet', 'inception'], type=str,
                    help='type of the model')
parser.add_argument('--loss', default='F1',  choices=['bce', 'bcew','focal','focalw','F1'], type=str,
                    help='type of loss')


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

transform=dict() 
mean=[0.054813755064775954, 0.0808928726780973, 0.08367144133595689, 0.05226083561943362]
std=[0.15201123862047256, 0.14087982537762958, 0.139965362113942, 0.10123220339551285]
transform['train']=transforms.Compose(
    [
     transforms.RandomAffine(20,shear=20,resample=PIL.Image.BILINEAR),
     #transforms.RandomRotation(20),
     transforms.RandomResizedCrop(512),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean,std)
     ])
transform['val']=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean,std)
     ])


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
        self.colors=['blue','red','yellow','green'];
        self.mode='CMYK'
        
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
        img_l4=[]
        for j,color in enumerate(self.colors):
            image_dir=os.path.join(self.img_dir,img_id+'_'+color+'.png')
            img_l4.append(PIL.Image.open(image_dir))
        img=PIL.Image.merge(mode=self.mode,bands=img_l4)
        
        '''
        for j,color in enumerate(self.colors):
            image_dir=os.path.join(self.img_dir,img_id+'_'+color+'.png')
            image=imageio.imread(image_dir)
            im_tensor[j,:,:]=torch.tensor(image,dtype=torch.float,device="cpu")/256
        '''
        if self.transform is not None:
            im_tensor = self.transform(img)
        if self.size is not None:
            im_tensor=F.adaptive_avg_pool2d(im_tensor, self.size)    
        
        
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
Resnet
'''
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls as resnet_uls
from torchvision.models.resnet import Bottleneck,BasicBlock
#import collections
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

        return x
'''
Inception 
'''
from torchvision.models.inception import model_urls as inception_url
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
Focal loss to handle imbalance between foreground and background
from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
https://arxiv.org/pdf/1708.02002.pdf
'''
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, logits=True, reduce=True, pos_weight=None ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.pos_weight =pos_weight

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs,targets,
                    reduction='none',pos_weight=self.pos_weight)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets,reduction='none')
        pt = torch.exp(-BCE_loss).detach()
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class F1Loss(nn.Module):
    def __init__(self):
        super(F1Loss, self).__init__()
    def forward(self,output,y_true):
        y_pred=output.sigmoid()
        tp = torch.sum(y_true*y_pred,0)
        fp = torch.sum(y_pred,0)
        fn = torch.sum(y_true,0)
        epsilon=1e-8
        p = tp / (fp + epsilon)
        r = tp / (fn + epsilon)
        f1 = 2*p*r / (p+r+epsilon)
        f1[f1!=f1]=0 #set NaN to 0
        return 1 - torch.mean(f1)    
    
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
    repeat=[];#pos_weight=[];
    for i in range(NLABEL):
        rep=int(np.power(len(ids[0])/len(ids[i]),0.5))
        repeat.append(rep)
        #pos_weight.append(np.power((len(label_dict)-rep*len(ids[i]))/len(ids[i])/rep,0.3))
        
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
    pos=np.zeros(NLABEL)
    num_train=0
    for phase in ['train','val']:
        for im in image_sets[phase]:
            im_label=label_dict[im]
            
            label_array=np.array(im_label)
            im_repeat= np.max(repeat[label_array])
            
            if phase=='val':
                image_labels[phase].append((im,im_label))
            else:
                for i in range(im_repeat):
                    pos[label_array]+=1
                    num_train+=1
                    image_labels[phase].append((im,im_label))
                    
                    

    dataset={x: ProteinDataset(args.root,x,image_labels[x],transform=transform[x]) 
            for x in ['train', 'val']}
    dataloader={x: torch.utils.data.DataLoader(dataset[x],
            batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
            for x in ['train', 'val']}
    
    if args.model=='resnet':
        model = ResNet(BasicBlock, [3, 4, 6, 3])
        model_url=resnet_uls['resnet34']
        con1_name='conv1.weight'
    elif args.model=='inception':
        model = Inception3()
        model_url=inception_url['inception_v3_google']
        con1_name='Conv2d_1a_3x3.conv.weight'
        
        
    #dataset_sizes={x: len(dataset[x]) for x in ['train', 'val']}
    #model = VGG(make_layers(cfg[args.type], batch_norm=True))
    #model =SqueezeNet(version=1.1)
    if not args.checkpoint:
        pre_trained=model_zoo.load_url(model_url)
        con1_weight=pre_trained[con1_name]        
        dim=np.random.choice(3,1)[0]
        pre_trained[con1_name]=torch.cat((con1_weight,
                   con1_weight[:,dim,:,:].unsqueeze_(1)),1)
        pre_trained['fc.weight']=pre_trained['fc.weight'][:NLABEL,:]
        pre_trained['fc.bias']=pre_trained['fc.bias'][:NLABEL]   
        if args.model=='inception':
            for key in list(pre_trained.keys()):
                if key.startswith('Aux'):
                    del pre_trained[key]
        model.load_state_dict(pre_trained)
    
    
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
    
    print('train number:',num_train)
    print('repeat:',repeat)
    print('positives:',pos)
    if args.loss.endswith('w'):
        pos_weight=torch.tensor(np.power((num_train-pos)/pos,0.4)).float().cuda()
        print('loss weights: ',pos_weight)
    else:
        pos_weight=None
        print('without loss weights')
    
    if args.loss=='F1':
        criterion = F1Loss()
        print('F1 loss')
    elif args.loss.startswith('bce'):
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print('BCEloss')
    elif args.loss.startswith('focal'):    
        criterion = FocalLoss(gamma=1,pos_weight=pos_weight)
        print('Focal Loss')
        
        
    optimizer = optim.SGD(model.parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.3)
    #t00 = time.time()
    #best_F1=0.0
    for i in range(args.resume_epoch):
        scheduler.step()
    for epoch in range(args.resume_epoch,args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 5)
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            
            running_loss=0
            running_prec=0
            running_recall=0
            correct_class=torch.zeros(NLABEL,dtype=torch.int64)
            selected_class=torch.zeros(NLABEL,dtype=torch.int64)
            relevant_class=torch.zeros(NLABEL,dtype=torch.int64)
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
                propose=(outputs.sigmoid()>0.5)
                targets=targets.byte()
                
                correct_class+=torch.sum(propose*targets,0)
                selected_class+=torch.sum(propose,0)
                relevant_class+=torch.sum(targets,0)
                
                corrects= torch.sum(propose*targets,1).double().cpu().numpy()
                selected= torch.sum(propose,1).double().cpu().numpy()
                relevant= torch.sum(targets,1).double().cpu().numpy()
                
                for i,c in enumerate(corrects):
                    if c>0:
                        running_prec +=corrects[i]/selected[i]
                        running_recall+= corrects[i]/relevant[i]
                    
                average_loss = running_loss/num
                average_prec = running_prec/num
                average_recall = running_recall/num
                if running_prec==0 or running_recall==0:
                    average_F1=0
                else:
                    average_F1 = 2/(1/average_prec+1/average_recall)
                t02 = time.time()
                if num % (100*inputs.size(0))==0:
                    print('{} L: {:.4f} p: {:.4f} r: {:.4f} F1: {:.4f} Time: {:.4f}s'.format(
                            num,average_loss,average_prec,average_recall,average_F1,t02-t01))
            
            '''print precision and recall for each class'''
            print(phase)
            class_prec=(correct_class.double()/selected_class.double()*100).cpu().numpy()
            class_recall=(correct_class.double()/relevant_class.double()*100).cpu().numpy()
            print('c:'+''.join('{:5d}'.format(i) for i in range(NLABEL)))
            print('n:'+''.join('{:5d}'.format(i) for i in correct_class.cpu().numpy()))
            print('p:'+''.join('{:5.0f}'.format(i) for i in class_prec))
            print('r:'+''.join('{:5.0f}'.format(i) for i in class_recall))
            
            
            if phase == 'val':
                #best_F1 = average_F1
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),os.path.join(args.root,args.save_folder,'out_'+str(epoch+1)+'.pth'))
        print()
        
if __name__ == '__main__':
    main()        


