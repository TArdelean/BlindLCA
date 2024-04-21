import torch
import torch.nn.functional as F
from tqdm import tqdm

import op_utils
from contrastive_learning import compute_CL_features

from data.feature_provider import FeaturesFromFolder
from data.masks import MaskMemory
from fca import ScFCA as FCA

def post_alpha(alpha, blur_sigma, clear_border):
    # Postprocessing alphas (See Section 3.3)
    if blur_sigma:
        alpha = op_utils.blur(alpha, kernel_size=5, sigma=blur_sigma)
    if clear_border:
        alpha = op_utils.clear_borders(alpha, border=clear_border)
    return alpha

def compute_descriptor(features, alpha, tau, subtract_mean):
    # Compute descriptors D_i as per Equation (2)
    assert features.shape[0] == 1  # Implemented only for batch size 1
    alpha = F.softmax(alpha.flatten() / tau, dim=0).reshape(alpha.shape)
    if subtract_mean:
        features = op_utils.blur(features, kernel_size=7, sigma=2.0)
        features = features - torch.mean(features, dim=(-2, -1), keepdim=True)
    descriptor = torch.sum(features[0] * alpha[None], dim=(1, 2))
    return descriptor

class JustFCA:
    def __init__(self, name="JustFCA", patch_size=(7, 7), sigma_p=3.0, k_s=5, sigma_s=1.0, chunk_size=20,
                 scale_features=True, clear_border=5, blur_sigma=1.0, tau=0.002):
        self.name = name
        self.fca = FCA(patch_size, sigma_p=sigma_p, k_s=k_s, sigma_s=sigma_s, chunk_size=chunk_size)
        self.scale_features = scale_features
        self.clear_border = clear_border
        self.blur_sigma = blur_sigma
        self.tau = tau
        self.subtract_mean = True

    def apply_fca(self, features):
        if self.scale_features:
            features = op_utils.scale_features(features)
        return self.fca(features)

    def call(self, data_loader):
        for img, label, mask, hw, img_path, idx in tqdm(data_loader.dataset):
            features = data_loader.feature_provider.get([idx])
            with torch.inference_mode():
                alpha = self.apply_fca(features)
                alpha = post_alpha(alpha, self.blur_sigma, self.clear_border)
                d_i = compute_descriptor(features, alpha, self.tau, self.subtract_mean)

            yield alpha, d_i


class ResidualsFCA(JustFCA):
    def __init__(self, name="ResidualsFCA", patch_size=(7, 7), sigma_p=3.0, k_s=5, sigma_s=1.0, chunk_size=20,
                 scale_features=True, clear_border=5, blur_sigma=1.0, tau=0.0025,
                 other_tag='Wide_ResNet50_2_Weights_V1_6_512', subtract_mean=True):
        super().__init__(name, patch_size, sigma_p, k_s, sigma_s, chunk_size, scale_features, clear_border, blur_sigma,
                         tau)
        self.other_tag = other_tag
        self.subtract_mean = subtract_mean

    def call(self, data_loader):
        # The primary feature provider is VAE; the second contains the wide features
        second_provider = FeaturesFromFolder(fe_tag=self.other_tag,
                                             cache_dir=data_loader.feature_provider.cache_dir,
                                             save_in_memory=False,
                                             device=data_loader.feature_provider.device)
        second_provider.init(data_loader.dataset)
        for img, label, mask, hw, img_path, idx in tqdm(data_loader.dataset):
            residuals = data_loader.feature_provider.get([idx])
            features = second_provider.get([idx])
            with torch.inference_mode():
                alpha = self.apply_fca(residuals)
                alpha = post_alpha(alpha, self.blur_sigma, self.clear_border)
                d_i = compute_descriptor(features, alpha, self.tau, self.subtract_mean)

            yield alpha, d_i


class OurBLCA(ResidualsFCA):
    def __init__(self, name="OurBALC", patch_size=(7, 7), sigma_p=3.0, k_s=5, sigma_s=1.0, chunk_size=20,
                 scale_features=True, clear_border=5, blur_sigma=1.0, tau=0.0025,
                 contrast_tag='contrast', subtract_mean=True, tiff_path=None, contrast_it=None, contrast_ep=10):
        super(OurBLCA, self).__init__(name, patch_size, sigma_p, k_s, sigma_s, chunk_size, scale_features,
                                      clear_border, blur_sigma, tau, contrast_tag, subtract_mean)
        self.contrast_tag = contrast_tag
        self.tiff_path = tiff_path
        self.must_compute_alpha = self.tiff_path is None
        self.contrast_it = contrast_it
        self.contrast_ep = contrast_ep
        assert self.contrast_it is None or self.contrast_ep is None

    def load_masks(self, data_loader):
        if self.must_compute_alpha:
            alphas = []
            print("Computing anomaly scores")
            for img, label, mask, hw, img_path, idx in tqdm(data_loader.dataset):
                with torch.inference_mode():
                    residuals = data_loader.feature_provider.get([idx])
                    alpha = self.apply_fca(residuals)
                    alpha = post_alpha(alpha, self.blur_sigma, self.clear_border)
                    alphas.append(alpha)
            op_utils.save_alpha_tiff(alphas, data_loader.dataset)
            self.tiff_path = "logs/alpha_tiff"

        mask_memory = MaskMemory(self.tiff_path, data_loader.dataset, data_loader.feature_provider.device)
        return mask_memory

    def generate(self, data_loader):
        cache_dir = data_loader.feature_provider.cache_dir
        masks = self.load_masks(data_loader)
        op_utils.set_seed(42)  # Legacy
        compute_CL_features(data_loader.dataset, masks, self.contrast_tag, cache_dir=cache_dir,
                            significant_tau=self.tau, iterations=self.contrast_it, epochs=self.contrast_ep)

    def call(self, data_loader):
        self.generate(data_loader)
        second_provider = FeaturesFromFolder(fe_tag=self.contrast_tag,
                                             cache_dir=data_loader.feature_provider.cache_dir,
                                             save_in_memory=False,
                                             device=data_loader.feature_provider.device)
        second_provider.init(data_loader.dataset)
        mask_memory = MaskMemory(self.tiff_path, data_loader.dataset, data_loader.feature_provider.device)

        for img, label, mask, hw, img_path, idx in tqdm(data_loader.dataset):
            cl_features = second_provider.get([idx])
            with torch.inference_mode():
                alpha = mask_memory.get(idx)[0]
                d_i = compute_descriptor(cl_features, alpha, self.tau, self.subtract_mean)

            yield alpha, d_i


class JustVAE:
    def __init__(self, name="JustVAE", clear_border=5, blur_sigma=1.0, tau=0.02,
                 other_tag='Wide_ResNet50_2_Weights_V1_6_512', subtract_mean=True, res_map='l1'):
        self.name = name
        self.clear_border = clear_border
        self.blur_sigma = blur_sigma
        self.tau = tau
        self.other_tag = other_tag
        self.subtract_mean = subtract_mean
        if res_map == 'l1':
            self.res_map = lambda f: torch.abs(f).mean(dim=(0, 1))
        elif res_map == 'l2':
            self.res_map = lambda f: torch.norm(f, p=2, dim=1).mean(dim=0)
        else:
            raise Exception("Invalid residual processor", res_map)

    def call(self, data_loader):
        # The primary feature provider is VAE; the second contains the wide features
        second_provider = FeaturesFromFolder(fe_tag=self.other_tag,
                                             cache_dir=data_loader.feature_provider.cache_dir,
                                             save_in_memory=False,
                                             device=data_loader.feature_provider.device)
        second_provider.init(data_loader.dataset)
        for img, label, mask, hw, img_path, idx in tqdm(data_loader.dataset):
            residuals = data_loader.feature_provider.get([idx])
            features = second_provider.get([idx])
            with torch.inference_mode():
                alpha = self.res_map(residuals)
                alpha = post_alpha(alpha, self.blur_sigma, self.clear_border)
                d_i = compute_descriptor(features, alpha, self.tau, self.subtract_mean)

            yield alpha, d_i
