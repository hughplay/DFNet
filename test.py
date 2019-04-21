from collections import defaultdict
from itertools import islice
from multiprocessing.pool import ThreadPool as Pool
import os
from pathlib import Path

import argparse
import cv2
import numpy as np
import torch
import tqdm

from utils import list2nparray, gen_miss, merge_imgs
from model import DFNet


class Tester:

    def __init__(self, model_path, input_size, batch_size):
        self.model_path = model_path
        self._input_size = input_size
        self.batch_size = batch_size
        self.init_model(model_path)

    @property
    def input_size(self):
        if self._input_size > 0:
            return (self._input_size, self._input_size)
        elif 'celeba' in self.model_path:
            return (256, 256)
        else:
            return (512, 512)

    def init_model(self, path):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using gpu.')
        else:
            self.device = torch.device('cpu')
            print('Using cpu.')

        self.model = DFNet().to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        print('Model %s loaded.' % path)

    def get_name(self, path):
        return '.'.join(path.name.split('.')[:-1])

    def results_path(self, output, img_path, mask_path, prefix='result'):
        img_name = self.get_name(img_path)
        mask_name = self.get_name(mask_path)
        return {
            'result_path': self.sub_dir('result').joinpath(
                'result-{}-{}.png'.format(img_name, mask_name)),
            'raw_path': self.sub_dir('raw').joinpath(
                'raw-{}-{}.png'.format(img_name, mask_name)),
            'alpha_path': self.sub_dir('alpha').joinpath(
                'alpha-{}-{}.png'.format(img_name, mask_name))
        }

    def inpaint_instance(self, img, mask):
        """Assume color image with 3 dimension. CWH"""
        img = img.view(1, *img.shape)
        mask = mask.view(1, 1, *mask.shape)
        return self.inpaint_batch(img, mask).squeeze()

    def inpaint_batch(self, imgs, masks):
        """Assume color channel is BGR and input is NWHC np.uint8."""
        imgs = np.transpose(imgs, [0, 3, 1, 2])
        masks = np.transpose(masks, [0, 3, 1, 2])

        imgs = torch.from_numpy(imgs).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)
        imgs = imgs.float().div(255)
        masks = masks.float().div(255)
        imgs_miss = imgs * masks
        results = self.model(imgs_miss, masks)
        if type(results) is list:
            results = results[0]
        results = results.mul(255).byte().data.cpu().numpy()
        results = np.transpose(results, [0, 2, 3, 1])
        return results

    def _process_file(self, output, img_path, mask_path):
        item = {
            'img_path': img_path,
            'mask_path': mask_path,
        }
        item.update(self.results_path(output, img_path, mask_path))
        self.path_pair.append(item)

    def process_single_file(self, output, img_path, mask_path):
        self.path_pair = []
        self._process_file(output, img_path, mask_path)

    def process_dir(self, output, img_dir, mask_dir):
        img_dir = Path(img_dir)
        mask_dir = Path(mask_dir)
        imgs_path = sorted(
            list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        masks_path = sorted(
            list(mask_dir.glob('*.jpg')) + list(mask_dir.glob('*.png')))

        n_img = len(imgs_path)
        n_mask = len(masks_path)
        n_pair = min(n_img, n_mask)

        self.path_pair = []
        for i in range(n_pair):
            img_path = imgs_path[i % n_img]
            mask_path = masks_path[i % n_mask]
            self._process_file(output, img_path, mask_path)

    def get_process(self, input_size):
        def process(pair):
            img = cv2.imread(str(pair['img_path']), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(pair['mask_path']), cv2.IMREAD_GRAYSCALE)
            if input_size:
                img = cv2.resize(img, input_size)
                mask = cv2.resize(mask, input_size)
            img = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.uint8)
            mask = np.ascontiguousarray(
                np.expand_dims(mask, 0)).astype(np.uint8)

            pair['img'] = img
            pair['mask'] = mask
            return pair
        return process

    def _file_batch(self):
        pool = Pool()

        n_pair = len(self.path_pair)
        n_batch = (n_pair-1) // self.batch_size + 1

        for i in tqdm.trange(n_batch, leave=False):
            _buffer = defaultdict(list)
            start = i * self.batch_size
            stop = start + self.batch_size
            process = self.get_process(self.input_size)
            batch = pool.imap_unordered(
                process, islice(self.path_pair, start, stop))
            for instance in batch:
                for k, v in instance.items():
                    _buffer[k].append(v)
            yield _buffer

    def batch_generator(self):
        generator = self._file_batch

        for _buffer in generator():
            for key in _buffer:
                if key in ['img', 'mask']:
                    _buffer[key] = list2nparray(_buffer[key])
            yield _buffer

    def to_numpy(self, tensor):
        tensor = tensor.mul(255).byte().data.cpu().numpy()
        tensor = np.transpose(tensor, [0, 2, 3, 1])
        return tensor

    def process_batch(self, batch, output):
        imgs = torch.from_numpy(batch['img']).to(self.device)
        masks = torch.from_numpy(batch['mask']).to(self.device)
        imgs = imgs.float().div(255)
        masks = masks.float().div(255)
        imgs_miss = imgs * masks

        result, alpha, raw = self.model(imgs_miss, masks)
        result, alpha, raw = result[0], alpha[0], raw[0]
        result = imgs * masks + result * (1 - masks)

        result = self.to_numpy(result)
        alpha = self.to_numpy(alpha)
        raw = self.to_numpy(raw)

        for i in range(result.shape[0]):
            cv2.imwrite(str(batch['result_path'][i]), result[i])
            cv2.imwrite(str(batch['raw_path'][i]), raw[i])
            cv2.imwrite(str(batch['alpha_path'][i]), alpha[i])

    @property
    def root(self):
        return Path(self.output)

    def sub_dir(self, sub):
        return self.root.joinpath(sub)

    def prepare_folders(self, folders):
        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)

    def inpaint(self, output, img, mask, merge_result=False):

        self.output = output
        self.prepare_folders([
            self.sub_dir('result'), self.sub_dir('alpha'),
            self.sub_dir('raw')])

        if os.path.isfile(img) and os.path.isfile(mask):
            if img.endswith(('.png', '.jpg', '.jpeg')):
                self.process_single_file(output, img, mask)
                _type = 'file'
            else:
                raise NotImplementedError()
        elif os.path.isdir(img) and os.path.isdir(mask):
            self.process_dir(output, img, mask)
            _type = 'dir'
        else:
            print('Img: ', img)
            print('Mask: ', mask)
            raise NotImplementedError(
                'img and mask should be both file or directory.')

        print('# Inpainting...')
        print('Input size:', self.input_size)
        for batch in self.batch_generator():
            self.process_batch(batch, output)
        print('Inpainting finished.')

        if merge_result and _type == 'dir':
            miss = self.sub_dir('miss')
            merge = self.sub_dir('merge')

            print('# Preparing input images...')
            gen_miss(img, mask, miss)
            print('# Merging...')
            merge_imgs([
                miss, self.sub_dir('raw'), self.sub_dir('alpha'),
                self.sub_dir('result'), img], merge, res=self.input_size[0])
            print('Merging finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', default='./model/model_places2.pth',
        help='Select a checkpoint.')
    parser.add_argument(
        '-i', '--input_size', default=0, type=int,
        help='Batch size for testing.')
    parser.add_argument(
        '-b', '--batch_size', default=8, type=int,
        help='Batch size for testing.')
    parser.add_argument(
        '--img', default='./samples/places2/img',
        help='Image or Image folder.')
    parser.add_argument(
        '--mask', default='./samples/places2/mask',
        help='Mask or Mask folder.')
    parser.add_argument('--output', default='./output/places2',
        help='Output dir')
    parser.add_argument(
        '--merge', action='store_true',
        help='Whether merge input and results for better viewing.')

    args = parser.parse_args()
    tester = Tester(args.model, args.input_size, args.batch_size)

    tester.inpaint(args.output, args.img, args.mask, merge_result=args.merge)
