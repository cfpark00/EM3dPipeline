import argparse

parser = argparse.ArgumentParser(description='Trains a Unet from a h5 dataset with on the fly augmentation')
parser.add_argument('train_dataset_h5', help='train dataset h5 path')
parser.add_argument('model_folder', help='model folder path')
parser.add_argument('-val_dataset_h5',type=str,default="", help='val dataset h5 path')
parser.add_argument('-n_steps', type=int,default=40000, help='number of gradient steps')
parser.add_argument('-save_every', type=int,default=1000, help='save every # gradient steps')
parser.add_argument('-n_val', type=int,default=100, help='number of validation batches')
parser.add_argument('-batch_size', type=int,default=16, help='batch size')
parser.add_argument('-lr', type=float,default=0.01, help='learning rate')

parser.add_argument('-pretrain_path',type=str,default="", help='pretrained network path')
parser.add_argument('-no_pad',action="store_true",help='No sub patching')
parser.add_argument('-dwt_sel',type=str,default="biased_fast", help='dwt selection for training data')

parser.add_argument('-N',type=int,default=2, help='number of CBR in each UNet block')
parser.add_argument('-width',type=int,default=32, help='width of UNet, 2*width will be the out channel of the first convolution')
parser.add_argument('-skip',action="store_true", help='add skip connections to each unet block')
parser.add_argument('-skipcat',action="store_true", help='skip connections concatenate and not add')
parser.add_argument('-catorig',action="store_true", help='concatenate the original input before the last convolution')
parser.add_argument('-outker',type=int,default=1, help='kernel size of the out convolution.')

args=parser.parse_args()

import torch
import numpy as np
import os
import glob
import tqdm
import pickle
device="cuda" if torch.cuda.is_available() else "cpu"

from src.NNtools import Dataset
from src.NNtools import UNet

train_dataset_h5=args.train_dataset_h5
model_folder=args.model_folder
val_dataset_h5=args.val_dataset_h5
n_steps=args.n_steps
save_every=args.save_every
n_val=args.n_val
batch_size=args.batch_size
lr=args.lr

pretrain_path=args.pretrain_path
no_pad=args.no_pad
dwt_sel=args.dwt_sel

N=args.N
width=args.width
skip=args.skip
skipcat=args.skipcat
catorig=args.catorig
outker=args.outker


if dwt_sel=="biased_fast":
    p_from_dwt_biased=lambda x: 1/x
elif dwt_sel=="biased_slow":
    p_from_dwt_biased=lambda x: x
elif dwt_sel=="only_slow":
    p_from_dwt_biased=lambda x: int(x>=800)
elif dwt_sel=="only_fast":
    p_from_dwt_biased=lambda x: int(x<=150)
elif dwt_sel=="unbiased":
    p_from_dwt_biased=lambda x: 1
else:
    assert False, "dwtsel not recognized"
p_from_dwt_unbiased=lambda x: 1

train_dataset=Dataset.PatchAugmentDataset(train_dataset_h5,n_steps*batch_size,p_from_dwt_biased,p_from_dwt_unbiased,do_pad=not no_pad)
if val_dataset_h5=="":
    val_dataset_h5=train_dataset_h5
val_dataset=Dataset.PatchAugmentDataset(val_dataset_h5,n_val*batch_size,p_from_dwt_biased,p_from_dwt_unbiased,do_pad=not no_pad)

def worker_init_fn(worker_id):
    np.random.seed()
    return

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,drop_last=True,num_workers=4,worker_init_fn=worker_init_fn)

net=UNet.UNet(n_channels=1,n_classes=2)
net=net.to(device=device,dtype=torch.float32)
if pretrain_path!="":
    net.load_state_dict(torch.load(pretrain_path))

opt=torch.optim.Adam(net.parameters(),lr=lr)
lossfunc=torch.nn.CrossEntropyLoss()

train_losses=[]
t_vals=[]
val_losses=[]

net.train()
step=0
pbar=tqdm.tqdm(total=n_steps,position=0,leave=True)
for frame,mask in train_loader:
    frame=frame.to(device=device,dtype=torch.float32)
    mask=mask.to(device=device,dtype=torch.int64)
    pred=net(frame)
    loss=lossfunc(pred,mask)
    opt.zero_grad()
    loss.backward()
    opt.step()
    train_losses.append(loss.item())
    step+=1
    pbar.update(1)
    
    if step%save_every==0:
        net.eval()
        val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,drop_last=True,num_workers=4,worker_init_fn=worker_init_fn)
        val_loss_avg=0
        c_val=0
        with torch.no_grad():
            for val_frame,val_mask in val_loader:
                val_frame=val_frame.to(device=device,dtype=torch.float32)
                val_mask=val_mask.to(device=device,dtype=torch.int64)
                pred=net(val_frame)
                loss=lossfunc(pred,val_mask)
                val_loss_avg+=loss.item()
                c_val+=1
        val_loss_avg=val_loss_avg/c_val
        t_vals.append(step)
        val_losses.append(val_loss_avg)
        torch.save(net.state_dict(),os.path.join(model_folder,"unet_{save_n:d}_{val_loss:.2e}.pth".format(save_n=step//save_every,val_loss=val_loss_avg)))
        
        net.train()
pbar.close()

logs={}
for arg in vars(args):
    key,val=arg,getattr(args, arg)
    logs[key]=val
logs["train_losses"]=train_losses
logs["t_vals"]=t_vals
logs["val_losses"]=val_losses

with open(os.path.join(model_folder,"logs.pkl"),"wb") as f:
    pickle.dump(logs,f)

print("Train succesful")
