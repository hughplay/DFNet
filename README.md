# Deep Fusion Network for Image completion

## Introduction

Deep image completion usually fails to harmonically blend the restored image into existing content,
especially in the boundary area. And it often fails to complete complex structures.

We first introduce **Fusion Block** for generating a flexible alpha composition map to combine known and unknown regions.
It builds a bridge for structural and texture information, so that information in known region can be naturally propagated into completion area.
With this technology, the completion results will have smooth transition near the boundary of completion area.

Furthermore, the architecture of fusion block enable us to apply **multi-scale constraints**.
Multi-scale constrains improves the performance of DFNet a lot on structure consistency.

Moreover, **it is easy to apply this fusion block and multi-scale constrains to other existing deep image completion models**.
A fusion block feed with feature maps and input image, will give you a completion result in the same resolution as given feature maps.

More detail can be found in our [paper](https://dl.acm.org/citation.cfm?doid=3343031.3351002)

The illustration of a fusion block:

<p align="center">
  <img width="600" src="imgs/fusion-block.jpg">
</p>

Examples of corresponding images:

![](imgs/github_teaser.jpg)

If you find this code useful for your research, please cite:

```
@inproceedings{Hong:2019:DFN:3343031.3351002,
 author = {Hong, Xin and Xiong, Pengfei and Ji, Renhe and Fan, Haoqiang},
 title = {Deep Fusion Network for Image Completion},
 booktitle = {Proceedings of the 27th ACM International Conference on Multimedia},
 series = {MM '19},
 year = {2019},
 isbn = {978-1-4503-6889-6},
 location = {Nice, France},
 pages = {2033--2042},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3343031.3351002},
 doi = {10.1145/3343031.3351002},
 acmid = {3351002},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {alpha composition, deep fusion network, fusion block, image completion, inpainting},
} 
```

## Prerequisites

- Python 3
- PyTorch 1.0
- OpenCV

## Testing

[<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/hughplay/DFNet/blob/master/demo.ipynb)


Clone this repo:

``` py
git clone https://github.com/hughplay/DFNet.git
cd DFNet
```

Download pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1lKJg__prvJTOdgmg9ZDF9II8B1C3YSkN?usp=sharing)
and put them into `model`.

### Testing with Places2 model

There are already some sample images in the `samples/places2` folder.

``` sh
python test.py --model model/model_places2.pth --img samples/places2/img --mask samples/places2/mask --output output/places2 --merge
```

### Testing with CelebA model

There are already some sample images in the `samples/celeba` folder.

``` sh
python test.py --model model/model_celeba.pth --img samples/celeba/img --mask samples/celeba/mask --output output/celeba --merge
```

## Training

Please refer to: https://github.com/deepcodebase/inpaint. It is building in progress but looks good so far.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

