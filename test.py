from train import ProteinDataset,VGG,make_layers,cfg,NAME,NLABEL
import os
#import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
import time

parser = argparse.ArgumentParser(
    description='Protain Alta Image Classification')
parser.add_argument('--root', default='./',
                    type=str, help='directory of the data')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--results', default='results/', type=str,
                    help='Dir to save results')
parser.add_argument('--checkpoint', default='A/A_38.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--type', default='A', choices=['A', 'B'], type=str,
                    help='type of the model')

args = parser.parse_args()
if not os.path.exists(args.results):
    os.mkdir(args.results)
    
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0")
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device("cpu")
def main():   
    csv_file=os.path.join(args.root,'sample_submission.csv')
    label_dict=pd.read_csv(csv_file, index_col=0, squeeze=True).to_dict()
    images= list(label_dict.keys())
    
    dataset=ProteinDataset(args.root,'test',images) 
    dataloader=torch.utils.data.DataLoader(dataset,
                batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    
    model = VGG(make_layers(cfg[args.type], batch_norm=True))
    
    if torch.cuda.is_available():
            model=nn.DataParallel(model)
            cudnn.benchmark = True
    
    if args.checkpoint:
        print('loading weights {}...'.format(args.checkpoint))
        weight_file=os.path.join(args.root,args.checkpoint)
        model.load_state_dict(torch.load(weight_file,
                                 map_location=lambda storage, loc: storage))
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    print('Finished loading model!')
    total=len(images)
    num=0
    output_file=os.path.join(args.root,args.results,args.type+'.txt')
    f=open( output_file , mode='w+')
    f.write('Id,Predicted\n')
    for inputs,_ in dataloader:
        t01 = time.time()
        inputs = inputs.to(device)
        outputs = model(inputs)
        propose=(outputs>0.5).cpu().numpy()
        for j in range(args.batch_size):
            predicts=list(propose[j,:].nonzero()[0])        
            image_id=images[num]
            num+=1
            f.write(image_id+', '+' '.join(str(label) for label in predicts)+'\n')
        t02= time.time()
        dt1=(t02-t01)/args.batch_size
        print('Image {:d}/{:d} time: {:.4f}s'.format(num+1,total,dt1))
    f.close()
    
if __name__ == '__main__':
    main()
        