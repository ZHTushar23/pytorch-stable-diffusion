'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu
'''


# Import Libraries
import os
import h5py
from torch.utils.data import Dataset, random_split, DataLoader
import torch
import numpy as np
import torchvision.transforms as T
from angleEmbedding import get_embedder  
# torch.manual_seed(0)
SZA= torch.tensor([60., 40., 20.,  4.])
VZA= torch.tensor([ 60,  30,  15,   0, -15, -30, -60])

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

class NasaDataset(Dataset):
    def __init__(self, root_dir,L=50):
        self.root_dir = root_dir
        self.filelist = [file for file in os.listdir(root_dir) if file.endswith(".hdf5")]
        # self.filelist = os.listdir(root_dir)
        # self.transform1 = T.Compose([T.ToTensor()])
        # self.transform2 = T.Compose([T.ToTensor()]) # define transformation
        self.embed, _ = get_embedder(L)
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        fname = self.root_dir+"/LES_profile_%05d.hdf5"%(idx+1)
        hf = h5py.File(fname, 'r')
        # print(hf.keys()) 

        # (Batch_Size, nsza, nvza, Channel, Height, Width)
        r_data = np.nan_to_num(np.array(hf.get('Reflectance_100m_resolution')))

        
        # cot profile
        # (Batch_Size, Channel, Height, Width)
        cot_data = np.empty((1,144,144), dtype=float) 
        cot_data[0,:,:] = np.nan_to_num(np.array(hf.get('Cloud_optical_thickness_(100m resolution)')))
        cot_data = np.log(cot_data+1)

        # # # SZA and VZA
        # SZA = np.nan_to_num(np.array(hf.get("SZA"))) 
        # VZA = np.nan_to_num(np.array(hf.get("VZA"))) 

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        sza_emb = self.embed(torch.tensor([SZA[3]]))
        vza_emb = self.embed(torch.tensor([VZA[3]]))
        context = torch.concat((sza_emb.unsqueeze(1),vza_emb.unsqueeze(1)),dim=1)

        # print("SZA: ",sza)
        # print("VZA: ",vza)
        
        # Cloud Mask
        # cmask = np.nan_to_num(np.array(hf.get("Native_Cloud_fraction_(100m)")))     
        hf.close()
        


        # Convert to tensor
        r_data = r_data[3,3,:,:,:]
        r_data = torch.tensor(r_data, dtype=torch.float32)
        cot_data = torch.tensor(cot_data, dtype=torch.float32)

        # rescale
        cot_data = rescale(cot_data,(0, 6), (-1, 1))

        sample = {'reflectance': r_data, 'cot': cot_data,'angles':context}
        return sample


if __name__=="__main__":
    # dataset_dir = "/nfs/rs/psanjay/users/ztushar1/COT_CER_Joint_Retrievals/one_thousand_profiles/Refl"
    dataset_dir = "/nfs/rs/psanjay/users/ztushar1/pytorch-stable-diffusion/data/LES_vers1_multiangle_results/"

    train_data = NasaDataset(root_dir=dataset_dir)
    loader = DataLoader(train_data, batch_size=10,shuffle=False)
    print(len(loader.dataset))
    temp= []
    temp1= []
    for i in range(len(loader.dataset)):
        data = loader.dataset[i]
        # get the data
        X, Y = data['reflectance'],data['cot']
        print(Y.shape, X.shape)
        break   
    #     x =    data['cot']
    #     temp.append(torch.max(x).item())
    #     temp1.append(torch.min(x).item())
    # print(np.max(temp),np.min(temp1))