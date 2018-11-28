from train import ProteinDataset,ResNet,BasicBlock,NAME,NLABEL,transform
import os
import numpy as np
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
parser.add_argument('--checkpoint', default='res34_1/out_24.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--model', default='resnet', type=str,
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
    label_dict=pd.read_csv(csv_file)
    images= []
    for i in range(len(label_dict)):
        name=label_dict.loc[i]['Id']
        images.append(name)
    
    dataset=ProteinDataset(args.root,'test',images,transform=transform['val']) 
    dataloader=torch.utils.data.DataLoader(dataset,
                batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    
    #model = VGG(make_layers(cfg[args.type], batch_norm=True))
    if args.model=='resnet':
        model = ResNet(BasicBlock, [3, 4, 6, 3])
    elif args.model=='inception':
        model = Inception3()
    
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
    output_file=os.path.join(args.root,args.results,'test_'+args.type+'.csv')
    f=open( output_file , mode='w+')
    f.write('Id,Predicted\n')
    t02=time.time()
    with torch.no_grad():
        selected_class=torch.zeros(NLABEL,dtype=torch.int64)
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            propose= (outputs.sigmoid()>0.5)
            selected_class+=torch.sum(propose,0)
            propose=propose.cpu().numpy()
            count=propose.shape[0]
            for j in range(count):
                image_id=images[num]
                predicts=list(propose[j,:].nonzero()[0]) 
                if len(predicts)==0:
                    predicts=np.argmax(propose[j,:])
                    f.write(image_id+','+str(predicts)+'\n')
                else:
                    predicts=sorted(predicts,key=lambda kv: NAME[kv])
                    f.write(image_id+','+' '.join(str(label) for label in predicts)+'\n')
                num+=1
            t01 = t02
            t02= time.time()
            dt1=(t02-t01)/count
            print('Image {:d}/{:d} time: {:.4f}s'.format(num,total,dt1))
    print('c:'+''.join('{:5d}'.format(i) for i in range(NLABEL)))
    print('n:'+''.join('{:5d}'.format(i) for i in selected_class.cpu().numpy()))
    f.close()
    
if __name__ == '__main__':
    main()
        