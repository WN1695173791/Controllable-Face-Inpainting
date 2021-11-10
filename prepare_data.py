
import os
import cv2
import lmdb
import glob
import argparse
import multiprocessing
import numpy as np
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from scipy.io import loadmat

from torchvision.transforms import functional as trans_fn

####################################################################################################
# run the code
# python prepare_data.py \
# --root /home/yurui/dataset/CelebA-HQ \
# --coeff_file /home/yurui/dataset/CelebA-HQ/semantic.mat \
# --dataset celeba \
# --out /home/yurui/dataset/CelebA-HQ
####################################################################################################

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(5)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


class Resizer:
    def __init__(self, *, size, coeff_3dmm, ext):
        self.size = size
        self.coeff_3dmm = coeff_3dmm

    def get_resized_bytes(self, img):
        img = trans_fn.resize(img, self.size, Image.BICUBIC)
        buf = BytesIO()
        img.save(buf, format='png')
        img_bytes = buf.getvalue()
        return img_bytes

    def get_3dmm_bytes(self, index):
        coeff_3dmm = self.coeff_3dmm[index:index+1]
        coeff_bytes = coeff_3dmm.tobytes()
        return coeff_bytes

    def prepare(self, filename, index):
        img = Image.open(filename)
        img = img.convert('RGB')
        img_bytes = self.get_resized_bytes(img)
        coeff_bytes = self.get_3dmm_bytes(index)
        return img_bytes, coeff_bytes

    def __call__(self, index_filename):
        index, filename = index_filename
        result_img, result_3dmm = self.prepare(filename, index)
        return index, result_img, result_3dmm, filename

def trans_3dmm(coeff_3dmm_dict):
    trans_param = coeff_3dmm_dict['transform_params']
    coeff = coeff_3dmm_dict['coeff']
    coeff_3dmm = np.concatenate([coeff, trans_param[:,2:]], 1)
    return coeff_3dmm


def prepare_data(root, coeff_file, dataset, out, n_worker, sizes, chunksize):
    assert dataset in ['celeba', 'ffhq']
    if dataset == 'ffhq':
        ext = 'png'
        filenames = sorted(glob.glob(f'{root}/*.{ext}'))
        coeff_3dmm = coeff_file
        coeff_3dmm = trans_3dmm(loadmat(coeff_3dmm))
    elif dataset == 'celeba':
        ext = 'jpg'
        filenames = sorted(glob.glob(f'{root}/*/images/*.{ext}'))
        coeff_3dmm = coeff_file
        coeff_3dmm = trans_3dmm(loadmat(coeff_3dmm))
        

    total = len(filenames)
    os.makedirs(out, exist_ok=True)

    for size in sizes:
        lmdb_path = os.path.join(out, str('-'.join([str(item) for item in size])))
        with lmdb.open(lmdb_path, map_size=1024 ** 4, readahead=False) as env:
            with env.begin(write=True) as txn:
                txn.put(format_for_lmdb('length'), format_for_lmdb(total))
                resizer = Resizer(size=size, coeff_3dmm=coeff_3dmm, ext=ext)
                with multiprocessing.Pool(n_worker) as pool:
                    for idx, result_img, result_coeff, filename in tqdm(
                            pool.imap_unordered(resizer, enumerate(filenames), chunksize=chunksize),
                            total=total):
                        filename = os.path.basename(filename)
                        filename = os.path.splitext(filename)[0]
                        txn.put(format_for_lmdb(filename), result_img)
                        filename = filename + '_3dmm'
                        txn.put(format_for_lmdb(filename), result_coeff)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, help='a path to output directory')
    parser.add_argument('--coeff_file', type=str, help='a path to output directory')

    parser.add_argument('--dataset', type=str, default='celeba', help='a path to output directory')
    parser.add_argument('--out', type=str, help='a path to output directory')
    # parser.add_argument('--sizes', type=int, nargs='+', default=((256, 176),(512, 352)))
    parser.add_argument('--sizes', type=int, nargs='+', default=((256, 256),) )
    parser.add_argument('--n_worker', type=int, help='number of worker processes', default=8)
    parser.add_argument('--chunksize', type=int, help='approximate chunksize for each worker', default=10)
    args = parser.parse_args()
    prepare_data(**vars(args))