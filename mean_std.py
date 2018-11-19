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
colors=['blue','red','yellow','green'];mode='CMYK'
mean=dict()
std=dict()
for color in colors:
    all_mean=0
    all_var=0
    for i,img in enumerate(label_dict.keys()):
        img_l4=[]
        image=os.path.join(image_dir,img+'_'+color+'.png')
            
        img4=PIL.Image.open(image)
        
        transform= transforms.Compose(
                    [transforms.ToTensor()])       
        img4t=transform(img4)     
        all_mean+=img4t.mean(2).mean(1).item()
        all_var+=img4t.var(2).var(1).item()
        if i%1000==0:
            print(i,len(label_dict))
    mean[color]=all_mean/len(label_dict)
    std[color]=np.sqrt(all_var/len(label_dict))
print(mean)
print(std)
'''dim=img4t.shape[0]
    if i==0:
        image_means =mean.view(1,dim)
        image_vars =mean.view(1,dim)
    else:
        image_means=torch.cat([image_means,mean.view(1,dim)],0)
        image_vars=torch.cat([image_vars,mean.view(1,dim)],0)
all_mean=image_means.mean(0)       
all_std=image_vars.mean(0).sqrt()  
print(all_mean,all_std)'''
