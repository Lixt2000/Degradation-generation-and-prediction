import numpy as np
import torch.utils.data as data
from logging import getLogger
import os
import pandas as pd
import torch
from pathlib import Path
from increment import increment, increment_multi

_GLOBAL_SEED = 0
logger = getLogger()


class GetDataLoader(data.Dataset):
    def __init__(self, ori_data, window_size=3, phase='train', data_split=[0.8, 0, 0.2], processed_dir='processed/', data_root='data/', file_name='test_data.npy', normalization='min_max', data_name='Inverse Gaussian'):
        self.root = data_root
        self.file_name = file_name
        self.normalization = normalization
        self.data_name = data_name
        self.input = self.load_data(ori_data, data_split, processed_dir, phase, window_size)
        self.n_data = len(self.input)


    def load_data(self, ori_data, data_split, processed_dir, phase, window_size):
        ########### input original data
        if self.data_name in ['wiener', 'gamma', 'Inverse Gaussian']:
            data_array = ori_data
            constant = data_array[:, 0]
            indices = [i for i in range(1, data_array.shape[1])]
            data_array = increment(data_array, indices)
        elif self.data_name == 'Fatigue':
            data_array = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/fatigue.xlsx', header=0,
                                     index_col=0)
            data_array = data_array.to_numpy()[1:, :11]
            constant = data_array[:, 0]
            indices = [i for i in range(1, data_array.shape[1])]
            data_array = increment(data_array, indices)
        elif self.data_name == 'Train wheel':
            data_array = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/trainwheel.xlsx', header=0)
            data_array = data_array.to_numpy()[:10, :]
            constant = data_array[:, 0]
            indices = [i for i in range(1, data_array.shape[1])]
            data_array = increment(data_array, indices)
        elif self.data_name == 'Laser':
            data_array = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/laserdata.xlsx', header=0)
            data_array = data_array.to_numpy()[:15, :]
            constant = data_array[:, 0]
            indices = [i for i in range(1, data_array.shape[1])]
            data_array = increment(data_array, indices)
        elif self.data_name == 'Interial navigition':
            data_array = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/interial navigition.xlsx',
                                     header=0, index_col=0)
            data_array = data_array.to_numpy()[:, :9]
            constant = data_array[:, 0]
            indices = [i for i in range(1, data_array.shape[1])]
            data_array = increment(data_array, indices)


        df = pd.DataFrame(data_array)
        features = df
        '''if self.data_name != 'Fatigue':
            features = df.drop(df.columns[0], axis=1)'''
        # save parameters
        torch.save(features.mean(), os.path.join(processed_dir, 'mean.pt'))
        torch.save(features.std(), os.path.join(processed_dir, 'std.pt'))
        torch.save(features.min(), os.path.join(processed_dir, 'min.pt'))
        torch.save(features.max(), os.path.join(processed_dir, 'max.pt'))

        if self.normalization == 'norm':
            features = (features - features.mean()) / features.std()
        elif self.normalization == 'min_max':
            features = (features - features.min()) / (features.max() - features.min() + 1e-5)

        # make directory
        # logger.info("processed directory is empty or missing, recreate!")
        # Path(processed_dir).mkdir(parents=True, exist_ok=True)
        len_x = len(features)

        train_x, valid_x, test_x = features[:int(len_x * data_split[0])], \
            features[int(len_x * data_split[0]): int(len_x * (1 - data_split[-1]))], \
            features[int(len_x * (1 - data_split[-1])):]

        # windows slide
        train_x, valid_x, test_x = self.window_split(train_x, window_size), \
                                    self.window_split(valid_x, window_size), \
                                    self.window_split(test_x, window_size)

        torch.save(train_x, os.path.join(processed_dir, 'train.pt'))
        torch.save(valid_x, os.path.join(processed_dir, 'valid.pt'))
        torch.save(test_x, os.path.join(processed_dir, 'test.pt'))
        logger.info("create data done!")

        if phase == 'train':
            return train_x
        elif phase == 'valid':
            return valid_x
        elif phase == 'test':
            return test_x
        else:
            raise 'wrong phase {}'.format(phase)

    def window_split(self, inputs_df, window_size):
        inputs = torch.tensor(inputs_df.values)
        inputs = inputs.unfold(1, window_size, 1)
        return inputs

    def __getitem__(self, item):
        import cv2
        cv2.setNumThreads(0)

        x = self.input[item]
        return x

    def __len__(self):
        return self.n_data


def make_dataloader(
    ori_data,
    batch_size,
    num_workers=8,
    shuffle=True,
    window_size=1,
    data_split=[0.8, 0, 0.2],
    processed_dir='data/processed/',
    normalization='min-max',
    data_name='Inverse Gaussian',
):
    #[0.8, 0, 0.2]
    # train data loader
    train_dataset = GetDataLoader(ori_data, phase='train', window_size=window_size, data_split=data_split, processed_dir=processed_dir, normalization=normalization, data_name=data_name)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle)

    # valid data loader
    valid_dataset = GetDataLoader(ori_data, phase='valid', window_size=window_size, data_split=data_split, processed_dir=processed_dir, normalization=normalization, data_name=data_name)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)

    # test data loader
    test_dataset = GetDataLoader(ori_data, phase='test', window_size=window_size, data_split=data_split, processed_dir=processed_dir, normalization=normalization, data_name=data_name)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)
    logger.info('dataset created')
    return train_data_loader, valid_data_loader, test_data_loader
