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
torch.manual_seed(0)

class NasaDataset(Dataset):
    """  Dataset types:
        1. 'cloud_25'
        2. 'cloud_50'
        3. 'cloud_75'
        4. 'cv_dataset'
        """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filelist = [file for file in os.listdir(root_dir) if file.endswith(".hdf5")]
        # self.filelist = os.listdir(root_dir)
        self.transform1 = T.Compose([T.ToTensor()])
        # self.transform2 = T.Compose([T.ToTensor()]) # define transformation
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        fname = self.root_dir+"/LES_profile_%05d.hdf5"%(idx+1)
        hf = h5py.File(fname, 'r')
        # print(hf.keys()) 

        r_data = np.empty((144,144,2), dtype=float) 
        temp              = np.nan_to_num(np.array(hf.get("Reflectance_100m_resolution")))
        # reflectance at 0.66 um
        r_data[:,:,0]   = temp[0,:,:]
        # reflectance at 2.13 um
        r_data[:,:,1]   = temp[1,:,:]
        # print(r_data[10,10,0])

        # # cot profile
        cot_data =np.empty((144,144,1),dtype=float) 
        cot_data[:,:,0] = np.nan_to_num(np.array(hf.get("Cloud_optical_thickness_(100m resolution)")))

        # # CER
        # cot_data[0,:,:,1] = np.nan_to_num(np.array(hf.get("CER_(100m resolution)")))

        # # SZA and VZA
        # sza = np.nan_to_num(np.array(hf.get("SZA"))) 
        # vza = np.nan_to_num(np.array(hf.get("VZA"))) 
        
        # Cloud Mask
        # cmask = np.nan_to_num(np.array(hf.get("Native_Cloud_fraction_(100m)")))     
        hf.close()
        


        # Convert to tensor
        r_data = self.transform1(r_data)
        # cmask         = self.transform1(cmask)
        cot_data = self.transform1(cot_data)
        # cot_data = self.transform2(cot_data)

        sample = {'reflectance': r_data, 'cot': cot_data}
        return sample


if __name__=="__main__":
    dataset_dir = "/nfs/rs/psanjay/users/ztushar1/COT_CER_Joint_Retrievals/one_thousand_profiles/Refl"

    train_data = NasaDataset(root_dir=dataset_dir)
    loader = DataLoader(train_data, batch_size=10,shuffle=False)
    for i in range(len(loader.dataset)):
        data = loader.dataset[i]
        # get the data
        X, Y = data['reflectance'],data['cot']
        print(Y.shape, X.shape)
        break  

    dataset_dir2 = "/nfs/rs/psanjay/users/ztushar1/COT_CER_Joint_Retrievals/ncer_fill3"

    test_data = NasaDataset(root_dir=dataset_dir2)
    loader = DataLoader(test_data, batch_size=10,shuffle=False)
    print(len(loader.dataset))
    for i in range(len(loader.dataset)):
        data = loader.dataset[i]
        # get the data
        X, Y = data['reflectance'],data['cot']
        print(Y.shape, X.shape)
        break  
    