
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
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of epochs to train')
args = parser.parse_args()

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
    def __init__(self,root,phase,image_labels, size=None ,transform=None):
        self.root=os.path.expanduser(root)
        self.phase=phase
        self.transform = transform
        self.size=size
        
        self.img_dir=os.path.join(root,'train_img')
        self.colors=['red','green','blue','yellow']
        self.image_labels=image_labels
        '''files=os.listdir(self.img_dir)
        for img in files:
            if img.endswith('blue.png'):
                ims=img.split('_')
                self.images+=[ims[0]]'''
        #csv_file=os.path.join(root,'train.csv')
        #self.labels=pd.read_csv(csv_file, index_col=0, squeeze=True).to_dict()
        #for key in self.labels.keys():
        #    self.labels[key]=[int(a) for a in self.labels[key].split(' ')]

    def __getitem__(self, index):
        if not self.phase in ['test']:
            img_id,labels=self.image_labels[index]
            target=torch.zeros((NLABEL),device="cpu")
            for label in labels:
                target[label]=1
        else:
            img_id=self.image_labels[index]
            target = None
        im_tensor=torch.zeros((len(self.colors),512,512),device="cpu")
        for j,color in enumerate(self.colors):
            image_dir=os.path.join(self.img_dir,img_id+'_'+color+'.png')
            image=imageio.imread(image_dir)
            im_tensor[j,:,:]=torch.tensor(image,dtype=torch.float,device="cpu")/256
        if self.size is not None:
            im_tensor=F.adaptive_avg_pool2d(im_tensor, self.size)    
        if self.transform is not None:
            im_tensor = self.transform(im_tensor)
        return (im_tensor,target)
        
    def __len__(self):
        return len(self.image_labels)

 
'''
custom-built vgg model
'''
cfg = {
    'A': [64,'M',128,'M',256,'M',384,'M',384,'M']
    }
def make_layers(cfg, batch_norm=True):
    layers =[nn.Conv2d(4, 32,kernel_size=7,stride=2,padding=3,bias=False),
             nn.BatchNorm2d(32),
             nn.ReLU(inplace=True)]
    in_channels = 32
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=NLABEL, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
                
        self.classifier = nn.Sequential(
            nn.Linear(384 * 8 * 8, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
            nn.Sigmoid(),
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
                
                

def main():
    csv_file=os.path.join(args.root,'train.csv')
    label_dict=pd.read_csv(csv_file, index_col=0, squeeze=True).to_dict()
    for key,label in label_dict.items():
        label_dict[key]=[int(a) for a in label.split(' ')]
        

    ids={i:[] for i in range(NLABEL)}
    for key,label in label_dict.items():
        for j in label:
            ids[j].append(key)
    
    image_labels={'train':[], 'val':[]}
    for l,ims in ids.items():
        ll=len(ims)
        vl=int(np.ceil(ll*0.1))
        larray=list(range(ll))
        varray=np.random.choice(ll, vl, replace=False)
        tarray=list(set(larray) - set(varray))
        for i in varray:
            im=ims[int(i)]
            image_labels['val'].append((im,label_dict[im]))
        for i in tarray:
            im=ims[int(i)]
            image_labels['train'].append((im,label_dict[im]))
    
    
    
    dataset={x: ProteinDataset(args.root,x,image_labels[x]) 
            for x in ['train', 'val']}
    dataloader={x: torch.utils.data.DataLoader(dataset[x],
            batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
            for x in ['train', 'val']}
    #dataset_sizes={x: len(dataset[x]) for x in ['train', 'val']}
    model = VGG(make_layers(cfg['A'], batch_norm=True))
    if torch.cuda.is_available():
        model=nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(),lr=args.lr, momentum=0.9, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.4)
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
                corrects= torch.sum(propose*targets,1).double()
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
                
                average_loss = running_loss/num
                average_F1 = running_F1/num
                t02 = time.time()
                if num % (100*inputs.size(0))==0:
                    print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'.format(
                            num, average_loss, average_F1, t02-t01))
            if phase == 'val' and average_F1 > best_F1:
                best_F1 = average_F1
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),os.path.join(args.root,'save','out_'+str(epoch)+'.pth'))
        print()
        
if __name__ == '__main__':
    main()        


