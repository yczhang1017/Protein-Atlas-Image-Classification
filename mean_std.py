import os
import PIL
import numpy as np
import pandas as pd
import scipy.misc
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import pickle

csv_file=os.path.join('./','train.csv')
label_dict=pd.read_csv(csv_file, index_col=0, squeeze=True).to_dict()
image_dir=os.path.join('./','train_img')
colors=['blue','red','yellow','green'];mode='CMYK'
mean=dict()
std=dict()
images=list(label_dict.keys())
randl=np.random.choice(len(images), int(len(images)/10))
for color in colors:
    all_mean=0
    all_var=0
    for i,ind in enumerate(randl):
        img=images[ind]
        img_l4=[]
        image=os.path.join(image_dir,img+'_'+color+'.png')
            
        img4=PIL.Image.open(image)
        
        transform= transforms.Compose(
                    [transforms.ToTensor()])       
        img4t=transform(img4)     
        all_mean+=img4t.mean().item()
        all_var+=img4t.var().item()
        if (i+1)%300==0:
            print(i,len(label_dict),all_mean/(i+1),all_var/(i+1))
    mean[color]=all_mean/len(randl)
    std[color]=np.sqrt(all_var/len(randl))
'''   
with open('mean_std.pkl', 'w') as f:
    pickle.dump(list(mean.values()), f)
    pickle.dump(list(std.values()), f)
'''
print(list(mean.values()))
print(list(std.values()))
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
