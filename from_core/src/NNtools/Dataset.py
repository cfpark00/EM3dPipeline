import torch
import glob
import os
import numpy as np
from PIL import Image
import scipy.ndimage as sim
import skimage.morphology as skmorph
import h5py
import tqdm

class PatchAugmentDataset(torch.utils.data.Dataset):
    def __init__(self,file_path,n_samples,p_from_dwt_biased,p_from_dwt_unbiased,do_pad=True):
        super().__init__()
        self.h5=h5py.File(file_path,"r")
        self.n_samples=n_samples
        
        self.W=self.h5.attrs["W"]
        self.H=self.h5.attrs["H"]
        
        self.dwts=self.h5.attrs["dwts"]
        self.regs=self.h5.attrs["regs"]
        self.p_dwts_biased=np.array([float(p_from_dwt_biased(dwt)) for dwt in self.dwts])
        self.p_dwts_biased/=self.p_dwts_biased.sum()
        self.p_dwts_unbiased=np.array([float(p_from_dwt_unbiased(dwt)) for dwt in self.dwts])
        self.p_dwts_unbiased/=self.p_dwts_unbiased.sum()
        
        self.im_dtype=None
        self.ims_masks={}
        with tqdm.tqdm(total=len(self.regs)*len(self.dwts)) as pbar:
            for reg in self.regs:
                for dwt in self.dwts:
                    im,mask=np.array(self.h5[reg+"/"+str(dwt)+"/im"]),np.array(self.h5[reg+"/"+str(dwt)+"/mask"])
                    self.ims_masks[(reg,dwt)]=im,mask
                    if self.im_dtype is None:
                        self.im_dtype=im.dtype
                    else:
                        assert im.dtype==self.im_dtype
                    pbar.update(1)
        self.h5.close()
        #print("dwts",self.dwts)
        #print("p_dwts_biased",self.p_dwts_biased)
        #print("p_dwts_unbiased",self.p_dwts_unbiased)
        self.patch_size=256
        self.pad_sizes=[5,10,20,40]
        self.p_seeds=[0.15,0.5]
        self.n_pads_per_patch=20 if do_pad else 0

        self.grid=np.stack(np.meshgrid(np.arange(self.patch_size),np.arange(self.patch_size),indexing="ij"),axis=0)-self.patch_size/2+0.5
        self.pad_grids={}
        for pad_size in self.pad_sizes:
            self.pad_grids[pad_size]=np.stack(np.meshgrid(np.arange(pad_size),np.arange(pad_size),indexing="ij"),axis=0)-pad_size/2+0.5
        self.out=int(np.sqrt(2)*(self.patch_size//2+1)+1)

    def random_shape_gen(self,grid,p_seed):
        pad=np.random.binomial(1,p_seed,grid.shape[1:])
        pad=skmorph.binary_dilation(pad,np.ones((3,3)))
        return pad

    def get_random_image_mask(self,p_dwts):
        reg=np.random.choice(self.regs)
        dwt=np.random.choice(self.dwts,p=p_dwts)
        im,mask=self.ims_masks[(reg,dwt)]
        return im,mask,reg,dwt
    
    def get_random_image_mask_from_reg(self,reg,p_dwts):
        dwt=np.random.choice(self.dwts,p=p_dwts)
        im,mask=self.ims_masks[(reg,dwt)]
        return im,mask,reg,dwt
    
    def __getitem__(self,i):
        if (not isinstance(i,int)) or i<0 or i>=self.n_samples:
            raise IndexError
        loc=self.out+np.array([np.random.choice(self.W-2*self.out),np.random.choice(self.H-2*self.out)])+np.random.random()-0.5
        theta=np.random.random()*2*np.pi
        rotmat=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        grid_=np.einsum("ij,jkm->ikm",rotmat,self.grid)
        if np.random.random()<0.5:
            grid_[0]*=-1
        grid_+=loc[:,None,None]
    
        im,mask,reg,dwt=self.get_random_image_mask(self.p_dwts_biased)
        im_=sim.map_coordinates(im,[grid_[0],grid_[1]],order=0)
        mask_=sim.map_coordinates(mask,[grid_[0],grid_[1]],order=0)
        for n_pad in range(self.n_pads_per_patch):
            pad_size=np.random.choice(self.pad_sizes)
            im_pad,_,_,_=self.get_random_image_mask_from_reg(reg,self.p_dwts_unbiased)
            im_pad_=sim.map_coordinates(im_pad,[grid_[0],grid_[1]],order=0)
            shape=self.random_shape_gen(self.pad_grids[pad_size],np.random.choice(self.p_seeds))
            loc=np.random.choice(self.patch_size,size=2)+np.random.random()-0.5
            ff=(loc[:,None]+self.pad_grids[pad_size][:,shape]).astype(np.int32)
            ff=ff[:,np.logical_and(np.logical_and(ff[0]>=0,ff[0]<self.patch_size),np.logical_and(ff[1]>=0,ff[1]<self.patch_size))]
            im_[ff[0],ff[1]]=im_pad_[ff[0],ff[1]]
        return torch.from_numpy(im_/np.iinfo(self.im_dtype).max)[None].to(dtype=torch.float32),torch.from_numpy(mask_/255).to(dtype=torch.int64)
    
    def __len__(self):
        return self.n_samples
