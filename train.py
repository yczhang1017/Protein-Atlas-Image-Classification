
import sys
import os
import numpy as np
import pandas as pd
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torch.autograd import Variable

dataset='./'
model='vgg16'
batch_size=16
workers=4
img_size = (256,256)


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
else:
    torch.set_default_tensor_type(torch.FloatTensor)

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, root='./',size=(256,256),transform=None,download=False):
        self.root=os.path.expanduser(root)
        self.transform = transform
        self.size=size
        self.colors=['red','green','blue','yellow']
        
        self.images=[]
        self.img_dir=os.path.join(root,'train_img')
        files=os.listdir(self.img_dir)
        for img in files:
            if img.endswith('blue.png'):
                ims=img.split('_')
                self.images+=[ims[0]]
        csv_file=os.path.join(root,'train.csv')
        self.labels=pd.read_csv(csv_file, index_col=0, squeeze=True).to_dict()
        for key in self.labels.keys():
            self.labels[key]=[int(a) for a in self.labels[key].split(' ')]

    def __getitem__(self, index):
        img_id=self.images[index]
        im_tensor=torch.zeros((len(self.colors),512,512))
        for j,color in enumerate(self.colors):
            image_dir=os.path.join(self.img_dir,img_id+'_'+color+'.png')
            image=imageio.imread(image_dir)
            im_tensor[j,:,:]=torch.tensor(image,dtype=torch.float)/256
            
        im_tensor=F.adaptive_avg_pool2d(im_tensor, self.size).permute(1,2,0)    
        if self.transform is not None:
            im_tensor = self.transform(im_tensor)
        if self.mode=='train':
            label=self.labels[img_id]
            labels=label.split(' ')
            return (img_id, im_tensor,labels)
        else:
            return (img_id,im_tensor)
    def __len__(self):
        return len(self.images)


PD=ProteinDataset(dataset)
all_labels=[]
for labels in PD.labels.values():
    for j in labels:
        all_labels.append(j)
from collections import Counter
c_val =Counter(all_labels)
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(1,1, figsize = (10, 5))
ax1.bar(range(NLABEL), [c_val[k] for k in range(NLABEL)])
for k in range(NLABEL):
    print(NAME[k], 'count:', c_val[k])
    
    
'''
csv_file=os.path.join(dataset,'train.csv')
image_df=pd.read_csv(csv_file)
image_df['target_list'] = image_df['Target'].map(lambda x: [int(a) for a in x.split(' ')])
#image_df=pd.read_csv(csv_file, header=None, index_col=0, squeeze=True)
image_df['target_list'] = image_df['Target'].map(lambda x: [int(a) for a in x.split(' ')])
'''
'''    
PD=ProteinDataset(dataset)
a=PD[2][1]
from matplotlib import pyplot as plt
plt.imshow(np.array(a[:,:,:3]))
'''




'''
if model=='vgg16':
    net=torchvision.models.vgg16()
    net.features[0]= nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=(1, 1))    
    net.classifier=nn.Sequential()
''' 


