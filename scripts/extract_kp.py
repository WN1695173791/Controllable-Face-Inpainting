import os
import time
import glob
import argparse
import face_alignment
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import cycle

from torch.multiprocessing import Pool, Process, set_start_method

class KeypointExtractor():
    def __init__(self):
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)   

    def extract_keypoints(self, filenames, outdir):
        assert isinstance(filenames, list)
        for filename in filenames:
            image = Image.open(filename)
            image = image.convert('RGB')

            current_kp = self.extract_keypoint(image)
            name = os.path.splitext(os.path.basename(filename))[0]

            if 'train' in filename:
                name = os.path.join('train', name)
            elif 'test' in filename:
                name = os.path.join('test', name)
            name = os.path.join(outdir, name.zfill(5)+'.txt')
            if not os.path.exists(os.path.dirname(name)):
                os.makedirs(os.path.dirname(name))
            np.savetxt(name, current_kp.reshape(-1))

    def extract_keypoint(self, image):
        while True:
            try:
                keypoints = self.detector.get_landmarks_from_image(np.array(image))[0]
                break
            except RuntimeError as e:
                if str(e).startswith('CUDA'):
                    print("Warning: out of memory, sleep for 1s")
                    time.sleep(1)
                else:
                    print(e)
                    break    
            except TypeError:
                print('No face detected in this image')
                shape = [68, 2]
                keypoints = -1. * np.ones(shape)                    
                break
        return keypoints



def run(data):
    filenames, opt, device = data
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    kp_extractor = KeypointExtractor()
    kp_extractor.extract_keypoints(filenames, opt.output_dir)


if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', type=str, help='the folder of the input files')
    parser.add_argument('--output_dir', type=str, help='the folder of the input files')
    parser.add_argument('--dataset', type=str, help='the folder of the output files')
    parser.add_argument('--device_ids', type=str, default='0,1')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--item_pre_processing', type=int, default=1000)

    opt = parser.parse_args()
    filenames = list()
    if opt.dataset == 'celeba':
        filenames = sorted(glob.glob(f'{opt.data_root}/**/images/*.jpg'))
    elif opt.dataset == 'ffhq':
        filenames = sorted(glob.glob(f'{opt.data_root}/*.png'))

    print('Total number of images:', len(filenames))

    all_filenames = []
    for i in range(len(filenames) // opt.item_pre_processing + 1):
        all_filenames.append(
            filenames[i*opt.item_pre_processing:(i+1)*opt.item_pre_processing]
        )

    pool = Pool(opt.workers)
    args_list = cycle([opt])
    device_ids = opt.device_ids.split(",")
    device_ids = cycle(device_ids)
    for data in tqdm(pool.imap_unordered(run, zip(all_filenames, args_list, device_ids))):
        None