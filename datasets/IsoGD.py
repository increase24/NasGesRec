import os
import glob
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, Image):
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class IsoGD(Dataset):
    def __init__(self, DataCfg, phase='train') -> None:
        super().__init__()
        self.phase = phase
        self.dataset_root = DataCfg.dataset_root
        annotation_file = os.path.join(DataCfg.dataset_root, f'{DataCfg.modality}_{phase}_list.txt')
        self.sample_duration = DataCfg.sample_duration
        self.resize_shape = DataCfg.resize_shape # (320, 240)
        self.crop_size = DataCfg.crop_size # 224
        self.flip_rate = DataCfg.flip_rate if phase=='train' else 0.0

        self.transform = transforms.Compose([
            Normaliztion(), 
            transforms.ToTensor()
        ])

        # select the samples with frame number over 8
        lines= filter(lambda x: x[1] > 8, self.__get_data_list_and_label(annotation_file))
        self.inputs = list(lines)

    def __get_data_list_and_label(self, annotation_file):
        return [(lambda arr: (arr[0], int(arr[1]), int(arr[2])))(i[:-1].split(' '))
                for i in open(annotation_file).readlines()]

    def transform_params(self, resize_shape=(320, 240), crop_size=224, flip_rate=0.5):
        if self.phase == 'train':
            left, top = random.randint(0, resize_shape[0] - crop_size), random.randint(0, resize_shape[1] - crop_size)
            is_flip = True if random.uniform(0, 1) < flip_rate else False
        else:
            left, top = (resize_shape[0] - crop_size) // 2, (resize_shape[1] - crop_size) // 2
            is_flip = False
        return (left, top, left + crop_size, top + crop_size), is_flip

    def __preprocessing(self, img, resize_shape, crop_rect, is_flip):
        img = img.resize(resize_shape)
        img = img.crop(crop_rect)
        if is_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return np.array(img.resize((112, 112)))

    def Sample_Image(self, imgs_path, frame_count, resize_shape, crop_rect, is_flip):
        frame_paths = sorted(glob.glob(os.path.join(imgs_path, '*.jpg')))
        sn = self.sample_duration
        if self.phase == 'train':
            f = lambda n: [(lambda n, arr: n if arr == [] else random.choice(arr))(
                n * i / sn,
                range( int(n * i / sn), max(int(n * i / sn) + 1, int(n * (i + 1) / sn)) )
                ) for i in range(sn)]
        else:
            f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(
                n * i / sn, 
                range(int(n * i / sn), max(int(n * i / sn) + 1, int(n * (i + 1) / sn)))
                ) for i in range(sn)]
        sl = f(frame_count)
        frames = []
        for idx in sl:
            img = self.__preprocessing(Image.open(frame_paths[idx]), resize_shape, crop_rect, is_flip) # RGB
            frames.append(self.transform(img).view(3, 112, 112, 1))
        return torch.cat(frames, dim=3).float()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        
        crop_rect, is_flip = self.transform_params(resize_shape=self.resize_shape, flip_rate = self.flip_rate)  # no flip
        data_path = os.path.join(self.dataset_root, self.inputs[index][0])
        clip = self.Sample_Image(data_path, self.inputs[index][1], self.resize_shape, crop_rect, is_flip)
        return clip.permute(0, 3, 1, 2), self.inputs[index][2] - 1

    def __len__(self):
        return len(self.inputs)


if __name__ == "__main__":
    from yacs.config import CfgNode as CN
    import matplotlib.pyplot as plt
    with open('./cfgs/IsoGD.yaml') as cfg_file:
        cfg = CN.load_cfg(cfg_file)
        print('Successfully loading the config file....')
    DataConfig = cfg.DatasetConfig
    trainset = IsoGD(DataConfig, 'train')
    sample = trainset[0]
    print('image sequence sample: ', sample[0].shape, '\ngesture label:', sample[1])
    frame_num = cfg.DatasetConfig.sample_duration
    plt.figure(figsize=(24,8))
    for idx in range(frame_num):
        image = sample[0][:,idx,:,:]
        image = (image * 128 + 127.5)/255 # convert [-1, 1] to [0, 1]
        image = image.permute(1,2,0).numpy()
        #image = image[:,:,::-1] # no need for 'Image.open'
        plt.subplot(frame_num//8, 8, idx+1)
        plt.imshow(image)
    plt.suptitle('IsoGD gesture video sample')
    plt.show()

