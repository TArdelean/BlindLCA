import random
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
import torchvision
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import contingency_matrix


def scale_features(tensor: torch.tensor) -> torch.tensor:
    # tensor shape B x C x H x W
    tf = tensor.flatten(start_dim=2)
    mini = tf.min(dim=-1).values[..., None, None]
    maxi = tf.max(dim=-1).values[..., None, None]
    div = (maxi - mini + 1e-8)
    return (tensor - mini) / div


# noinspection PyUnresolvedReferences
def blur(image, kernel_size=7, sigma=None):
    if sigma is None:
        sigma = kernel_size / 4
    shape = image.shape
    im_b = image[(None,) * (4 - len(shape))]
    return torchvision.transforms.functional.gaussian_blur(im_b, kernel_size, sigma=sigma).view(shape)

def reflect_pad(image, patch_size=7):
    p = patch_size // 2
    return torch.nn.functional.pad(image, (p, p, p, p), mode='reflect')

def get_gaussian_kernel(device, tile_size, s: float):
    return torchvision.transforms._functional_tensor._get_gaussian_kernel2d(tile_size, [s, s], torch.float32, device)


def clear_borders(alphas, border=5):
    alphas[..., :border, :] = 0
    alphas[..., -border:, :] = 0
    alphas[..., :, :border] = 0
    alphas[..., :, -border:] = 0
    return alphas


def save_clusters(dataset, labels):
    path_tokens = [dataset.path_tokens(path) for path in dataset.img_paths]
    obj_names, class_names, img_names = zip(*path_tokens)
    df = pd.DataFrame({'object': obj_names,
                       'class': class_names,
                       'img_name:': img_names,
                       'label': labels})
    df.to_csv('img_assignment.csv', mode='a')


def compute_cluster_f1(gt_labels, pred_labels, average='micro'):
    # Compute the contingency matrix
    contingency = contingency_matrix(gt_labels, pred_labels)

    # Use Hungarian matching to find the best cluster assignments
    row_ind, col_ind = linear_sum_assignment(-contingency)

    contingency[:, row_ind] = contingency[:, col_ind]
    print(contingency)

    # Reassign the predicted labels based on the best cluster assignments
    pred_labels_mapped = np.zeros_like(pred_labels)
    for i, j in zip(row_ind, col_ind):
        pred_labels_mapped[pred_labels == j] = i

    return f1_score(gt_labels, pred_labels_mapped, average=average)


def save_alpha_unit(alphas, dataset):
    print("Saving alphas normalized per unit")
    root_dir = Path("logs/alpha_unit")
    for alpha, (_, _, _, _, img_path, idx) in zip(alphas, dataset):
        obj_name, class_name, img_name = dataset.path_tokens(img_path)
        out_path = root_dir / obj_name / class_name
        out_path.mkdir(parents=True, exist_ok=True)
        alpha = scale_features(alpha[None, None])[0, 0]
        gray = (alpha.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(gray).save(out_path / img_name.replace('png', 'jpg'))


def save_alpha_tiff(alphas, dataset):
    print("Saving alphas in tiff format for evaluation")
    root_dir = Path("logs/alpha_tiff")
    for alpha, (_, _, _, _, img_path, idx) in zip(alphas, dataset):
        obj_name, class_name, img_name = dataset.path_tokens(img_path)
        img_name = img_name.split('.')[0]
        out_path = root_dir / obj_name / "test" / class_name
        out_path.mkdir(parents=True, exist_ok=True)
        np_alpha = alpha.cpu().numpy()
        tiff.imsave(out_path / f"{img_name}.tiff", np_alpha)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
