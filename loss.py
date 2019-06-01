import torch
from torch import nn
from torchvision import models

from utils import resize_like


class ReconstructionLoss(nn.L1Loss):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, results, targets):
        loss = 0.
        for i, (res, target) in enumerate(zip(results, targets)):
            loss += self.l1(res, target)
        return loss / len(results)


class VGGFeature(nn.Module):

    def __init__(self):
        super().__init__()

        vgg16 = models.vgg16(pretrained=True)
        for para in vgg16.parameters():
            para.requires_grad = False

        self.vgg16_pool_1 = nn.Sequential(*vgg16.features[0:5])
        self.vgg16_pool_2 = nn.Sequential(*vgg16.features[5:10])
        self.vgg16_pool_3 = nn.Sequential(*vgg16.features[10:17])

    def forward(self, x):

        pool_1 = self.vgg16_pool_1(x)
        pool_2 = self.vgg16_pool_2(pool_1)
        pool_3 = self.vgg16_pool_3(pool_2)

        return [pool_1, pool_2, pool_3]


class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, vgg_results, vgg_targets):
        loss = 0.
        for i, (vgg_res, vgg_target) in enumerate(
                zip(vgg_results, vgg_targets)):
            for feat_res, feat_target in zip(vgg_res, vgg_target):
                loss += self.l1loss(feat_res, feat_target)
        return loss / len(vgg_results)


class StyleLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def gram(self, feature):
        n, c, h, w = feature.shape
        feature = feature.view(n, c, h * w)
        gram_mat = torch.bmm(feature, torch.transpose(feature, 1, 2))
        return gram_mat / (c * h * w)

    def forward(self, vgg_results, vgg_targets):
        loss = 0.
        for i, (vgg_res, vgg_target) in enumerate(
                zip(vgg_results, vgg_targets)):
            for feat_res, feat_target in zip(vgg_res, vgg_target):
                loss += self.l1loss(
                    self.gram(feat_res), self.gram(feat_target))
        return loss / len(vgg_results)


class TotalVariationLoss(nn.Module):

    def __init__(self, c_img=3):
        super().__init__()
        self.c_img = c_img

        kernel = torch.FloatTensor([
            [0, 1, 0],
            [1, -2, 0],
            [0, 0, 0]]).view(1, 1, 3, 3)
        kernel = torch.cat([kernel] * c_img, dim=0)
        self.register_buffer('kernel', kernel)

    def gradient(self, x):
        return nn.functional.conv2d(
            x, self.kernel, stride=1, padding=1, groups=self.c_img)

    def forward(self, results, mask):
        loss = 0.
        for i, res in enumerate(results):
            grad = self.gradient(res) * resize_like(mask, res)
            loss += torch.mean(torch.abs(grad))
        return loss / len(results)


class InpaintLoss(nn.Module):

    def __init__(
            self, c_img=3, w_l1=6., w_percep=0.1, w_style=240., w_tv=0.1,
            structure_layers=[0, 1, 2, 3, 4, 5],
            texture_layers=[0, 1, 2]):

        super().__init__()

        self.l_struct = structure_layers
        self.l_text = texture_layers

        self.w_l1 = w_l1
        self.w_percep = w_percep
        self.w_style = w_style
        self.w_tv = w_tv

        self.reconstruction_loss = ReconstructionLoss()

        self.vgg_feature = VGGFeature()
        self.style_loss = StyleLoss()
        self.perceptual_loss = PerceptualLoss()
        self.tv_loss = TotalVariationLoss(c_img)

    def forward(self, results, target, mask):

        targets = [resize_like(target, res) for res in results]

        loss_struct = 0.
        loss_text = 0.
        loss_list = {}

        if len(self.l_struct) > 0:

            struct_r = [results[i] for i in self.l_struct]
            struct_t = [targets[i] for i in self.l_struct]

            loss_struct = self.reconstruction_loss(
                struct_r, struct_t) * self.w_l1

            loss_list['reconstruction_loss'] = loss_struct.item()

        if len(self.l_text) > 0:

            text_r = [targets[i] for i in self.l_text]
            text_t = [results[i] for i in self.l_text]

            vgg_r = [self.vgg_feature(f) for f in text_r]
            vgg_t = [self.vgg_feature(t) for t in text_t]

            loss_style = self.style_loss(vgg_r, vgg_t) * self.w_style
            loss_percep = self.perceptual_loss(vgg_r, vgg_t) * self.w_percep
            loss_tv = self.tv_loss(text_r, mask) * self.w_tv

            loss_text = loss_style + loss_percep + loss_tv
            loss_list.update({
                'perceptual_loss': loss_percep.item(),
                'style_loss': loss_style.item(),
                'total_variation_loss': loss_tv.item()
            })

        loss_total = loss_struct + loss_text

        return loss_total, loss_list
