import os
import lmdb
import random
import numpy as np
from io import BytesIO
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict

from data.mask_generator import RandomMask


class BaseDataset(Dataset):
    def __init__(self, opt, is_inference):
        path = opt.path

        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.image_index = self.get_image_index(is_inference)


        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = opt.resolution
        self.semantic_dim = 257 

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])

    def __len__(self):
        return len(self.image_index)

    def __getitem__(self, index):
        data={}
        index = self.image_index[index]
        with self.env.begin(write=False) as txn:
            key = f'{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key) 
            key_3dmm = f'{str(index).zfill(5)}_3dmm'.encode('utf-8')
            coeff_bytes = txn.get(key_3dmm) 

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        coeff_3dmm_np = np.frombuffer(coeff_bytes, dtype=np.float32)
        coeff_3dmm, crop_affine = self.transform_semantic(
            coeff_3dmm_np.reshape(1,260))
        mask = torch.from_numpy(RandomMask(self.resolution))
        
        data['mask']=mask
        data['coeff_3dmm']=coeff_3dmm
        data['crop_affine']=crop_affine
        data['input_image']=img
        data['key']=f'{str(index).zfill(5)}.png'
        return data


    def transform_semantic(self, semantic):
        coeff_3dmm = torch.from_numpy(semantic.astype(np.float32))[0, :257]
        crop_param = semantic[:,257:260]
        w0, h0 = 1024, 1024
        ratio, t0, t1 = np.hsplit(crop_param.astype(np.float32), 3)
        resolution = self.resolution
        ratio = ratio/256.*1024

        w = resolution * ratio
        h = resolution * ratio
        t0, t1 = float(t0)/w0*resolution, float(t1)/h0*resolution
        left = (w/2 - resolution/2 + (t0 - resolution/2)*ratio) * 1/ratio
        up = (h/2 - resolution/2 + (resolution/2 - t1)*ratio) * 1/ratio
        left = (left - (1-1/ratio)*resolution*0.5)/resolution*2.0
        up = (up - (1-1/ratio)*resolution*0.5)/resolution*2.0

        affine = np.float32([
            [1/ratio,0,left],
            [0,1/ratio,up],
            [0,0,1],])
        affine_inv = torch.from_numpy(np.linalg.inv(affine).astype(np.float32))
        return coeff_3dmm.squeeze(0), affine_inv

    def get_minibatch_val_np(self, batch_size):
        items = random.sample(range(self.__len__()), batch_size)
        return_data = defaultdict(list)
        for index in items:
            data = self.__getitem__(index)
            for item in data:
                return_data[item].append(data[item])

        for item in return_data:
            if torch.is_tensor(return_data[item][0]):
                return_data[item] = torch.cat(
                    [tensor[None] for tensor in return_data[item]], 0)
        return return_data

    def get_image_index(self, is_inference):
        raise NotImplementedError