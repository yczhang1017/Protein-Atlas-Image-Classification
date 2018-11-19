import os
import PIL
import numpy as np
import pandas as pd
import scipy.misc
from matplotlib import pyplot as plt
from torchvision import transforms
import torch

csv_file=os.path.join('./','train.csv')
label_dict=pd.read_csv(csv_file, index_col=0, squeeze=True).to_dict()
image_dir=os.path.join('./','train_img')
colors=['blue','yellow','red','green'];mode='CMYK'
for i,img in enumerate(label_dict.keys()):
        print(img)
        img_l4=[]
        for j,color in enumerate(colors):
            image=os.path.join(image_dir,img+'_'+color+'.png')
            img_l4.append(PIL.Image.open(image))
        img4=PIL.Image.merge(mode=mode,bands=img_l4)
        
        transform= transforms.Compose(
                    [transforms.ToTensor()])       
        img4t=transform(img4)     
        mean=img4t.mean(2).mean(1)
        var=img4t.var(2).var(1)**2
        dim=img4t.shape[0]
        if i==0:
            image_means =mean.view(1,dim)
            image_vars =mean.view(1,dim)
        else:
            image_means=torch.cat([image_means,mean.view(1,dim)],0)
            image_vars=torch.cat([image_vars,mean.view(1,dim)],0)

all_mean=image_means.mean(0)       
all_std=image_vars.mean(0).sqrt()  
print(all_mean,all_std)