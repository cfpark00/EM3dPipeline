import torch
import glob
import os
import numpy as np
from PIL import Image
import scipy.ndimage as sim
import skimage.morphology as skmorph
import h5py
import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self,basedir,subfol=False):
        self.basedir=basedir
        self.frame_fol=os.path.join(self.basedir,"frames")
        self.mask_fol=os.path.join(self.basedir,"masks")
        self.subfol=subfol
        if self.subfol:
            self.mask_files=glob.glob(os.path.join(self.mask_fol,"*","*"))
        else:
            self.mask_files=glob.glob(os.path.join(self.mask_fol,"*"))
        self.n=len(self.mask_files)

    def __getitem__(self,i):
        mask_file=self.mask_files[i]
        if self.subfol:
            mask_subfol,mask_name=os.path.split(mask_file)
            mask_subfol=os.path.split(mask_subfol)[1]
            frame_file=os.path.join(self.frame_fol,mask_subfol,mask_name)
        else:
            frame_file=os.path.join(self.frame_fol,os.path.split(mask_file)[1])
        frame=np.array(Image.open(frame_file))/255
        frame=torch.from_numpy(frame).unsqueeze(0).to(dtype=torch.float32)#add channel dimension
        mask=(np.array(Image.open(mask_file))/255)
        mask=torch.from_numpy(mask).to(dtype=torch.int64)
        return frame,mask

    def get_file_path(self,i):
        mask_file=self.mask_files[i]
        if self.subfol:
            mask_subfol,mask_name=os.path.split(mask_file)
            mask_subfol=os.path.split(mask_subfol)[1]
            frame_file=os.path.join(self.frame_fol,mask_subfol,mask_name)
        else:
            frame_file=os.path.join(self.frame_fol,os.path.split(mask_file)[1])
        return frame_file,mask_file

    def __len__(self):
        return self.n

class DatasetNerveRing(torch.utils.data.Dataset):
    def __init__(self,basedir,subfol=False):
        self.basedir=basedir
        self.frame_fol=os.path.join(self.basedir,"frames")
        self.mask_fol=os.path.join(self.basedir,"masks")
        self.subfol=subfol
        if self.subfol:
            self.mask_files=glob.glob(os.path.join(self.mask_fol,"*","*"))
        else:
            self.mask_files=glob.glob(os.path.join(self.mask_fol,"*"))
        self.n=len(self.mask_files)

    def __getitem__(self,i):
        mask_file=self.mask_files[i]
        if self.subfol:
            mask_subfol,mask_name=os.path.split(mask_file)
            mask_subfol=os.path.split(mask_subfol)[1]
            frame_file=os.path.join(self.frame_fol,mask_subfol,mask_name)
        else:
            frame_file=os.path.join(self.frame_fol,os.path.split(mask_file)[1])
        frame=np.array(Image.open(frame_file))/255
        frame=torch.from_numpy(frame).unsqueeze(0).to(dtype=torch.float32)#add channel dimension
        mask=(np.array(Image.open(mask_file))/255)
        mask=torch.from_numpy(mask).to(dtype=torch.int64)
        return frame,mask

    def get_file_path(self,i):
        mask_file=self.mask_files[i]
        if self.subfol:
            mask_subfol,mask_name=os.path.split(mask_file)
            mask_subfol=os.path.split(mask_subfol)[1]
            frame_file=os.path.join(self.frame_fol,mask_subfol,mask_name)
        else:
            frame_file=os.path.join(self.frame_fol,os.path.split(mask_file)[1])
        return frame_file,mask_file

    def __len__(self):
        return self.n
    
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

class PatchAugmentDatasetError(torch.utils.data.Dataset):
    def __init__(self,file_path,n_samples,p_from_dwt_biased):
        super().__init__()
        self.h5=h5py.File(file_path,"r")
        self.n_samples=n_samples
        
        self.W=self.h5.attrs["W"]
        self.H=self.h5.attrs["H"]
        
        self.dwts=self.h5.attrs["dwts"]
        self.regs=self.h5.attrs["regs"]
        self.p_dwts_biased=np.array([float(p_from_dwt_biased(dwt)) for dwt in self.dwts])
        self.p_dwts_biased/=self.p_dwts_biased.sum()
        
        self.im_dtype=None
        self.error_dtype=None
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
                    if self.error_dtype is None:
                        self.error_dtype=mask.dtype
                    else:
                        assert mask.dtype==self.error_dtype
                    pbar.update(1)
        self.h5.close()
        self.patch_size=256

        self.grid=np.stack(np.meshgrid(np.arange(self.patch_size),np.arange(self.patch_size),indexing="ij"),axis=0)-self.patch_size/2+0.5
        self.out=int(np.sqrt(2)*(self.patch_size//2+1)+1)

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
        return torch.from_numpy(im_/np.iinfo(self.im_dtype).max)[None].to(dtype=torch.float32),torch.from_numpy(mask_/np.iinfo(self.error_dtype).max)[None].to(dtype=torch.float32)
    
    def __len__(self):
        return self.n_samples
    
class PatchAugmentDatasetRealError(torch.utils.data.Dataset):
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
                    rescan_map=np.array(self.h5[reg+"/"+str(dwt)+"/rescan_map"])
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
        rescan_map_=sim.map_coordinates(np.array(self.h5[reg+"/"+str(dwt)+"/rescan_map"]),[grid_[0],grid_[1]],order=0)
        #
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

class PatchAugmentDatasetNerveRing(torch.utils.data.Dataset):
    def __init__(self,file_path,n_samples,p_from_dwt_biased):
        super().__init__()
        self.h5=h5py.File(file_path,"r")
        self.n_samples=n_samples
        
        self.dwts=self.h5.attrs["dwts"]
        self.regs=self.h5.attrs["regs"]
        self.p_dwts_biased=np.array([float(p_from_dwt_biased(dwt)) for dwt in self.dwts])
        self.p_dwts_biased/=self.p_dwts_biased.sum()
        
        self.im_dtype=None
        self.error_dtype=None
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
                    if self.error_dtype is None:
                        self.error_dtype=mask.dtype
                    else:
                        assert mask.dtype==self.error_dtype
                    pbar.update(1)
        self.h5.close()
        self.patch_size=256

    def get_random_image_mask(self,p_dwts):
        reg=np.random.choice(self.regs)
        dwt=np.random.choice(self.dwts,p=p_dwts)
        im,mask=self.ims_masks[(reg,dwt)]
        return im,mask,reg,dwt
 
    def sample(self,im,mask,patch_size=256):
        assert im.shape==mask.shape
        #randomly sample a NxN patch from the image and mask with any rotation and flip
        rotation=np.random.random()*2*np.pi
        N_big=int(patch_size*np.sqrt(2)+1)
        x=np.random.randint(0,im.shape[0]-N_big)
        y=np.random.randint(0,im.shape[1]-N_big)
        im_=im[x:x+N_big,y:y+N_big]
        mask_=mask[x:x+N_big,y:y+N_big]
        im_=sim.rotate(im_,rotation,reshape=False,order=1)
        mask_=sim.rotate(mask_,rotation,reshape=False,order=0)
        if np.random.random()<0.5:
            im_=np.flip(im_,axis=0)
            mask_=np.flip(mask_,axis=0)
        #get center for shape NxN
        x=(im_.shape[0]-patch_size)//2
        y=(im_.shape[1]-patch_size)//2
        return im_[x:x+patch_size,y:y+patch_size].copy(),mask_[x:x+patch_size,y:y+patch_size].copy()
    
    def __getitem__(self,i):
        if (not isinstance(i,int)) or i<0 or i>=self.n_samples:
            raise IndexError
        im,mask,reg,dwt=self.get_random_image_mask(self.p_dwts_biased)
        im,mask=self.sample(im,mask,self.patch_size)
        return torch.from_numpy(im/np.iinfo(self.im_dtype).max)[None].to(dtype=torch.float32),torch.from_numpy(mask/np.iinfo(self.error_dtype).max).to(dtype=torch.int64)
    
    def __len__(self):
        return self.n_samples
    
class PatchAugmentDatasetFocus(torch.utils.data.Dataset):
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
        
        self.patch_size=256
        self.pad_sizes=[5,10,20,40]
        self.p_seeds=[0.15,0.5]
        self.n_pads_per_patch=20 if do_pad else 0
        
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


        self.grid=np.stack(np.meshgrid(np.arange(self.patch_size),np.arange(self.patch_size),indexing="ij"),axis=0)-self.patch_size/2+0.5
        self.pad_grids={}
        for pad_size in self.pad_sizes:
            self.pad_grids[pad_size]=np.stack(np.meshgrid(np.arange(pad_size),np.arange(pad_size),indexing="ij"),axis=0)-pad_size/2+0.5
        self.out=int(np.sqrt(2)*(self.patch_size//2+1)+1)

    def random_shape_gen(self,grid,p_seed):
        pad=np.random.binomial(1,p_seed,grid.shape[1:])
        pad=sim.binary_dilation(pad,np.ones((3,3)))
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

    
"""
class PatchAugmentDatasetOnlySlowNoPatch(torch.utils.data.Dataset):
    def __init__(self,ims_masks,n_samples):
        super().__init__()
        self.ims_masks=ims_masks
        self.n_samples=n_samples
        
        self.W=None
        self.H=None
        
        self.regs=list(ims_masks.keys())
        self.dwts={}
        self.reps={}
        self.p_dwts_biased={100:0.,200:0.,400:0.,800:0.,1200:0.,1600:0.,2000:0.,2500:1}
        self.p_dwts_unbiased={100:0.,200:0.,400:0.,800:0.,1200:0.,1600:0.,2000:0.,2500:1.}
        for reg,reg_ims_masks in self.ims_masks.items():
            self.dwts[reg]=list(reg_ims_masks.keys())
            for dwt,reg_dwt_ims_masks in reg_ims_masks.items():
                self.reps[(reg,dwt)]=list(reg_dwt_ims_masks.keys())
                for rep,item in reg_dwt_ims_masks.items():
                    im,mask=item
                    if self.W is None:
                        self.W=im.shape[0]
                    else:
                        assert self.W==im.shape[0]
                    if self.H is None:
                        self.H=im.shape[1]
                    else:
                        assert self.H==im.shape[1]
                        
        self.patch_size=256
        self.pad_sizes=[5,10,20,40]
        self.p_seeds=[0.15,0.5]
        self.n_pads_per_patch=0

        self.grid=np.stack(np.meshgrid(np.arange(self.patch_size),np.arange(self.patch_size),indexing="ij"),axis=0)-self.patch_size/2+0.5
        self.pad_grids={}
        for pad_size in self.pad_sizes:
            self.pad_grids[pad_size]=np.stack(np.meshgrid(np.arange(pad_size),np.arange(pad_size),indexing="ij"),axis=0)-pad_size/2+0.5
        self.out=int(np.sqrt(2)*(self.patch_size//2+1)+1)

        def random_shape_gen(grid,p_seed):
            pad=np.random.binomial(1,p_seed,grid.shape[1:])
            pad=sim.binary_dilation(pad,np.ones((3,3)))
            return pad
        self.random_shape_gen=random_shape_gen

    def get_random_image_mask(self,p_dwts):
        reg=np.random.choice(self.regs)
        dwts=self.dwts[reg]
        p=np.array([p_dwts[dwt] for dwt in dwts])
        p/=p.sum()
        dwt=np.random.choice(dwts,p=p)
        rep=np.random.choice(self.reps[(reg,dwt)])
        im,mask=self.ims_masks[reg][dwt][rep]
        return im,mask,reg,dwt,rep
    
    def get_random_image_mask_from_reg(self,reg,p_dwts):
        dwts=self.dwts[reg]
        p=np.array([p_dwts[dwt] for dwt in dwts])
        p/=p.sum()
        dwt=np.random.choice(dwts,p=p)
        rep=np.random.choice(self.reps[(reg,dwt)])
        im,mask=self.ims_masks[reg][dwt][rep]
        return im,mask,reg,dwt,rep
    
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
    
        im,mask,reg,dwt,rep=self.get_random_image_mask(self.p_dwts_biased)
        im_=sim.map_coordinates(im,[grid_[0],grid_[1]],order=0)
        mask_=sim.map_coordinates(mask,[grid_[0],grid_[1]],order=0)
        for n_pad in range(self.n_pads_per_patch):
            pad_size=np.random.choice(self.pad_sizes)
            im_pad,_,_,_,_=self.get_random_image_mask_from_reg(reg,self.p_dwts_unbiased)
            im_pad_=sim.map_coordinates(im_pad,[grid_[0],grid_[1]],order=0)
            shape=self.random_shape_gen(self.pad_grids[pad_size],np.random.choice(self.p_seeds))
            loc=np.random.choice(self.patch_size,size=2)+np.random.random()-0.5
            ff=(loc[:,None]+self.pad_grids[pad_size][:,shape]).astype(np.int32)
            ff=ff[:,np.logical_and(np.logical_and(ff[0]>=0,ff[0]<self.patch_size),np.logical_and(ff[1]>=0,ff[1]<self.patch_size))]
            im_[ff[0],ff[1]]=im_pad_[ff[0],ff[1]]
        return torch.from_numpy(im_/255)[None].to(dtype=torch.float32),torch.from_numpy(mask_/255).to(dtype=torch.int64)
    
    def __len__(self):
        return self.n_samples

class PatchAugmentDatasetOnlyFastNoPatch(torch.utils.data.Dataset):
    def __init__(self,ims_masks,n_samples):
        super().__init__()
        self.ims_masks=ims_masks
        self.n_samples=n_samples
        
        self.W=None
        self.H=None
        
        self.regs=list(ims_masks.keys())
        self.dwts={}
        self.reps={}
        self.p_dwts_biased={100:1.,200:0.,400:0.,800:0.,1200:0.,1600:0.,2000:0.,2500:0.}
        self.p_dwts_unbiased={100:1.,200:0.,400:0.,800:0.,1200:0.,1600:0.,2000:0.,2500:0.}
        for reg,reg_ims_masks in self.ims_masks.items():
            self.dwts[reg]=list(reg_ims_masks.keys())
            for dwt,reg_dwt_ims_masks in reg_ims_masks.items():
                self.reps[(reg,dwt)]=list(reg_dwt_ims_masks.keys())
                for rep,item in reg_dwt_ims_masks.items():
                    im,mask=item
                    if self.W is None:
                        self.W=im.shape[0]
                    else:
                        assert self.W==im.shape[0]
                    if self.H is None:
                        self.H=im.shape[1]
                    else:
                        assert self.H==im.shape[1]
                        
        self.patch_size=256
        self.pad_sizes=[5,10,20,40]
        self.p_seeds=[0.15,0.5]
        self.n_pads_per_patch=0

        self.grid=np.stack(np.meshgrid(np.arange(self.patch_size),np.arange(self.patch_size),indexing="ij"),axis=0)-self.patch_size/2+0.5
        self.pad_grids={}
        for pad_size in self.pad_sizes:
            self.pad_grids[pad_size]=np.stack(np.meshgrid(np.arange(pad_size),np.arange(pad_size),indexing="ij"),axis=0)-pad_size/2+0.5
        self.out=int(np.sqrt(2)*(self.patch_size//2+1)+1)

        def random_shape_gen(grid,p_seed):
            pad=np.random.binomial(1,p_seed,grid.shape[1:])
            pad=sim.binary_dilation(pad,np.ones((3,3)))
            return pad
        self.random_shape_gen=random_shape_gen

    def get_random_image_mask(self,p_dwts):
        reg=np.random.choice(self.regs)
        dwts=self.dwts[reg]
        p=np.array([p_dwts[dwt] for dwt in dwts])
        p/=p.sum()
        dwt=np.random.choice(dwts,p=p)
        rep=np.random.choice(self.reps[(reg,dwt)])
        im,mask=self.ims_masks[reg][dwt][rep]
        return im,mask,reg,dwt,rep
    
    def get_random_image_mask_from_reg(self,reg,p_dwts):
        dwts=self.dwts[reg]
        p=np.array([p_dwts[dwt] for dwt in dwts])
        p/=p.sum()
        dwt=np.random.choice(dwts,p=p)
        rep=np.random.choice(self.reps[(reg,dwt)])
        im,mask=self.ims_masks[reg][dwt][rep]
        return im,mask,reg,dwt,rep
    
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
    
        im,mask,reg,dwt,rep=self.get_random_image_mask(self.p_dwts_biased)
        im_=sim.map_coordinates(im,[grid_[0],grid_[1]],order=0)
        mask_=sim.map_coordinates(mask,[grid_[0],grid_[1]],order=0)
        for n_pad in range(self.n_pads_per_patch):
            pad_size=np.random.choice(self.pad_sizes)
            im_pad,_,_,_,_=self.get_random_image_mask_from_reg(reg,self.p_dwts_unbiased)
            im_pad_=sim.map_coordinates(im_pad,[grid_[0],grid_[1]],order=0)
            shape=self.random_shape_gen(self.pad_grids[pad_size],np.random.choice(self.p_seeds))
            loc=np.random.choice(self.patch_size,size=2)+np.random.random()-0.5
            ff=(loc[:,None]+self.pad_grids[pad_size][:,shape]).astype(np.int32)
            ff=ff[:,np.logical_and(np.logical_and(ff[0]>=0,ff[0]<self.patch_size),np.logical_and(ff[1]>=0,ff[1]<self.patch_size))]
            im_[ff[0],ff[1]]=im_pad_[ff[0],ff[1]]
        return torch.from_numpy(im_/255)[None].to(dtype=torch.float32),torch.from_numpy(mask_/255).to(dtype=torch.int64)
    
    def __len__(self):
        return self.n_samples
    
class PatchAugmentDataset(torch.utils.data.Dataset):
    def __init__(self,ims_masks,n_samples):
        super().__init__()
        self.ims_masks=ims_masks
        self.n_samples=n_samples
        
        self.W=None
        self.H=None
        
        self.regs=list(ims_masks.keys())
        self.dwts={}
        self.reps={}
        self.p_dwts_biased={100:1.,200:0.2,400:0.2,800:0.2,1200:0.1,1600:0.1,2000:0.1,2500:0.1}
        self.p_dwts_unbiased={100:1.,200:1.,400:1.,800:1.,1200:1.,1600:1.,2000:1.,2500:1.}
        for reg,reg_ims_masks in self.ims_masks.items():
            self.dwts[reg]=list(reg_ims_masks.keys())
            for dwt,reg_dwt_ims_masks in reg_ims_masks.items():
                self.reps[(reg,dwt)]=list(reg_dwt_ims_masks.keys())
                #print(reg,dwt,list(reg_dwt_ims_masks.keys()))
                for rep,item in reg_dwt_ims_masks.items():
                    im,mask=item
                    if self.W is None:
                        self.W=im.shape[0]
                    else:
                        assert self.W==im.shape[0]
                    if self.H is None:
                        self.H=im.shape[1]
                    else:
                        assert self.H==im.shape[1]
                        
        self.patch_size=256
        self.pad_sizes=[5,10,20,40]
        self.p_seeds=[0.15,0.5]
        self.n_pads_per_patch=30

        self.grid=np.stack(np.meshgrid(np.arange(self.patch_size),np.arange(self.patch_size),indexing="ij"),axis=0)-self.patch_size/2+0.5
        self.pad_grids={}
        for pad_size in self.pad_sizes:
            self.pad_grids[pad_size]=np.stack(np.meshgrid(np.arange(pad_size),np.arange(pad_size),indexing="ij"),axis=0)-pad_size/2+0.5
        self.out=int(np.sqrt(2)*(self.patch_size//2+1)+1)

        def random_shape_gen(grid,p_seed):
            pad=np.random.binomial(1,p_seed,grid.shape[1:])
            pad=sim.binary_dilation(pad,np.ones((3,3)))
            return pad
        self.random_shape_gen=random_shape_gen

    def get_random_image_mask(self,p_dwts):
        reg=np.random.choice(self.regs)
        dwts=self.dwts[reg]
        p=np.array([p_dwts[dwt] for dwt in dwts])
        p/=p.sum()
        dwt=np.random.choice(dwts,p=p)
        rep=np.random.choice(self.reps[(reg,dwt)])
        im,mask=self.ims_masks[reg][dwt][rep]
        return im,mask,reg,dwt,rep
    
    def get_random_image_mask_from_reg(self,reg,p_dwts):
        dwts=self.dwts[reg]
        p=np.array([p_dwts[dwt] for dwt in dwts])
        p/=p.sum()
        dwt=np.random.choice(dwts,p=p)
        rep=np.random.choice(self.reps[(reg,dwt)])
        im,mask=self.ims_masks[reg][dwt][rep]
        return im,mask,reg,dwt,rep
    
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
    
        im,mask,reg,dwt,rep=self.get_random_image_mask(self.p_dwts_biased)
        im_=sim.map_coordinates(im,[grid_[0],grid_[1]],order=0)
        mask_=sim.map_coordinates(mask,[grid_[0],grid_[1]],order=0)
        for n_pad in range(self.n_pads_per_patch):
            pad_size=np.random.choice(self.pad_sizes)
            im_pad,_,_,_,_=self.get_random_image_mask_from_reg(reg,self.p_dwts_unbiased)
            im_pad_=sim.map_coordinates(im_pad,[grid_[0],grid_[1]],order=0)
            shape=self.random_shape_gen(self.pad_grids[pad_size],np.random.choice(self.p_seeds))
            loc=np.random.choice(self.patch_size,size=2)+np.random.random()-0.5
            ff=(loc[:,None]+self.pad_grids[pad_size][:,shape]).astype(np.int32)
            ff=ff[:,np.logical_and(np.logical_and(ff[0]>=0,ff[0]<self.patch_size),np.logical_and(ff[1]>=0,ff[1]<self.patch_size))]
            im_[ff[0],ff[1]]=im_pad_[ff[0],ff[1]]
        return torch.from_numpy(im_/255)[None].to(dtype=torch.float32),torch.from_numpy(mask_/255).to(dtype=torch.int64)
    
    def __len__(self):
        return self.n_samples

class ErrorPatchAugmentDataset(torch.utils.data.Dataset):
    def __init__(self,ims_masks,n_samples):
        super().__init__()
        self.ims_masks=ims_masks
        self.n_samples=n_samples
        
        self.W=None
        self.H=None
        
        self.keys=list(self.ims_masks.keys())
        for key,item in self.ims_masks.items():
            im,mask=item
            if self.W is None:
                self.W=im.shape[0]
            else:
                assert self.W==im.shape[0]
            if self.H is None:
                self.H=im.shape[1]
            else:
                assert self.H==im.shape[1]
                        
        self.patch_size=256
        self.grid=np.stack(np.meshgrid(np.arange(self.patch_size),np.arange(self.patch_size),indexing="ij"),axis=0)-self.patch_size/2+0.5
        self.out=int(np.sqrt(2)*(self.patch_size//2+1)+1)
    
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
    
        key=np.random.choice(self.keys)
        im,mask=self.ims_masks[key]
        im_=sim.map_coordinates(im,[grid_[0],grid_[1]],order=0)
        mask_=sim.map_coordinates(mask,[grid_[0],grid_[1]],order=0)
        return torch.from_numpy(im_/255)[None].to(dtype=torch.float32),torch.from_numpy(mask_/255).to(dtype=torch.int64)
    
    def __len__(self):
        return self.n_samples
"""
