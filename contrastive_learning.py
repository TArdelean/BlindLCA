from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import op_utils
from data import MVTecDataset
from data.feature_provider import WideFeatures
from data.masks import MaskMemory
from op_utils import blur
from sklearn import cluster


class ContrastNet(nn.Module):
    def __init__(self, latent=32, nf=256, in_dim=512):
        super(ContrastNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, nf, 1),
            nn.ReLU(),
            nn.Conv2d(nf, latent, 1),
        )

    def forward(self, x):
        return torch.nn.functional.normalize(self.net(x), dim=1)


def get_descriptor(features, mask, tau=0.002):
    assert features.ndim == 3
    features = blur(features, sigma=2.0)
    features = features - torch.mean(features, dim=(-2, -1), keepdim=True)
    features = features.flatten(start_dim=1)  # C x HW

    weights = torch.softmax(mask.flatten() / tau, dim=0)[None]  # 1 x HW
    descriptor = (features * weights).sum(dim=1)
    return descriptor


def compute_all_descriptors(mask_memory, fp, tau=0.002):
    all_significant = []
    for idx in range(len(mask_memory.memory)):
        features = fp.get(idx)
        ss_target = mask_memory.get(idx)
        all_significant.append(get_descriptor(features, ss_target, tau=tau))
    all_significant = torch.stack(all_significant)
    return all_significant


def get_neighbors(all_significant, k=3):
    my_sim = (1 - torch.square(all_significant[:, None, :] - all_significant[None, :, :]).mean(dim=-1) * 100)  # B x B

    top_indices = torch.argsort(my_sim, dim=1, descending=True)
    closest_neighbors = top_indices[:, 1:k + 1].detach().cpu().numpy()
    counter_neighbors = top_indices[:, top_indices.shape[0] // 2:].detach().cpu().numpy()
    return closest_neighbors, counter_neighbors


def select_random(tensor, size):
    if tensor.shape[0] == 0:
        return tensor
    perm = torch.randint(tensor.shape[0], (size,), device=tensor.device).view(-1)
    return tensor[perm]


def make_loss_margin(same_pos, same_neg, other_pos, other_neg, counter_pos, margin=0.5):
    def dist(t1, t2):
        # Note that vectors t1 and t2 are assumed here to be normalized
        return torch.square(t1[:, None] - t2[None, :]).sum(dim=-1).view(-1)

    attract = torch.cat([dist(same_pos, other_pos), dist(same_neg, other_neg)]).mean()
    repulse = torch.cat([torch.relu(margin - dist(same_pos, other_neg)),
                         torch.relu(margin - dist(other_pos, counter_pos))]).mean()
    return attract + repulse


def decide_thresholds(mask_memory, all_significant, dataset, method='heuristic'):
    if method == 'fixed':
        return lambda ind: (0.033, 0.033)
    elif method == "quant":
        def apply(ind):
            target = mask_memory.memory[ind]
            thr_q = torch.quantile(target.view(-1), 0.99).item()
            return thr_q, thr_q

        return apply
    elif method == "heuristic":
        # See Section S4 in the supplementary material
        cm = cluster.KMeans(n_clusters=dataset.label_ids.max() + 1, random_state=42)
        cm.fit(all_significant.cpu().numpy())
        lists = [[idx for idx in range(len(cm.labels_)) if cm.labels_[idx] == label_id] for label_id in
                 range(dataset.label_ids.max() + 1)]

        anomaly_scores = mask_memory.memory
        max_anomaly = torch.max(anomaly_scores.flatten(start_dim=1), dim=-1).values
        anomaly_per_label = torch.stack([max_anomaly[cli].mean() for cli in lists])
        good_idx = torch.argmin(anomaly_per_label).item()  # Predicted index for the 'good' label
        good_quantile = len(lists[good_idx]) / len(cm.labels_)
        thr = torch.quantile(max_anomaly, good_quantile).item()
        print("Threshold", thr)

        def apply(_):
            return thr, thr

        return apply
    else:
        raise Exception(f"Unknown method {method}")


def train_contrast_iteration(mask_memory, fp, who, net, closest_neighbors, counter_neighbors, th_method,
                             same_size=20, close_size=50, counters_num=5):
    def get_pos_neg(i, projector):
        thresholds = th_method(i)
        features = op_utils.scale_features(fp.get(i))
        f_projected = projector(features)  # B x C x H x W
        mask = mask_memory.get(i)
        mask = mask[:, 0, :, :]  # B x H x W
        f_projected = f_projected.permute(1, 0, 2, 3)  # C x B x H x W
        pos_mask = mask > thresholds[1]
        neg_mask = mask < thresholds[0]
        pos = f_projected[:, pos_mask].T  # N x C
        neg = f_projected[:, neg_mask].T  # M x C
        return pos, neg

    same = get_pos_neg([who], net)
    same_pos, same_neg = select_random(same[0], same_size), select_random(same[1], same_size)
    close = get_pos_neg(closest_neighbors[who], net)
    close = (torch.cat([close[0], same[0]]), torch.cat([close[1], same[1]]))
    close_pos = select_random(close[0], close_size)
    close_neg = select_random(close[1], close_size)  # B x D
    counters = get_pos_neg(np.random.choice(counter_neighbors[who], size=counters_num), net)
    counter_pos = select_random(counters[0], close_size)
    loss = make_loss_margin(same_pos, same_neg, close_pos, close_neg, counter_pos, margin=0.5)
    return loss


def compute_CL_features(dataset, mask_memory, tag, cache_dir='cache', k=3, significant_tau=0.002,
                        iterations=2000, epochs=None, device=torch.device('cuda:0')):
    print("Computing contrastive features for", dataset.object_name)

    net = ContrastNet(512).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0005, weight_decay=0.01)

    cache_path = Path(cache_dir) / dataset.data_root.name / tag
    fp = WideFeatures(device=device, save_in_memory=False, cache_dir=cache_dir)
    fp.init(dataset)
    all_significant = compute_all_descriptors(mask_memory, fp, tau=significant_tau)
    closest_neighbors, counter_neighbors = get_neighbors(all_significant, k=k)

    if epochs is None:
        epochs = (iterations - 1) // len(dataset) + 1

    ano = np.arange(len(dataset))
    thm = decide_thresholds(mask_memory, all_significant, dataset)
    for _ in tqdm(range(epochs)):
        for idx in np.random.permutation(ano):
            loss = train_contrast_iteration(mask_memory, fp, idx, net, closest_neighbors, counter_neighbors, thm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    dataset = dataset.__class__(object_name=dataset.object_name, resize=dataset.resize, split="test",
                                exclude_combined=False, data_root=dataset.data_root)
    fp.init(dataset)
    for idx, img_path in enumerate(dataset.img_paths):
        with torch.no_grad():
            features = fp.get([idx])
            features = op_utils.scale_features(features)
            projected = net(features)[0]

        obj_name, class_name, img_name = dataset.path_tokens(str(img_path))
        (cache_path / obj_name).mkdir(parents=True, exist_ok=True)
        assert obj_name == dataset.object_name
        file_path = cache_path / dataset.object_name / f"{class_name}_{img_name.split('.')[0]}.pt"
        torch.save(projected, file_path)
    (cache_path / "weights").mkdir(exist_ok=True)
    torch.save(net.state_dict(), str(cache_path / "weights" / f"{dataset.object_name}.pt"))


def generate_contrasted(tiff_path, tag, cache_dir='cache', dataset_class=MVTecDataset, iterations=20,
                        k=3, significant_tau=0.002, device=torch.device('cuda:0')):
    for object_name in dataset_class.objects:
        dataset = dataset_class(object_name=object_name, exclude_combined=True)
        mask_memory = MaskMemory(tiff_path, dataset, device)
        compute_CL_features(dataset, mask_memory, tag, cache_dir, k, significant_tau, iterations)


if __name__ == '__main__':
    op_utils.set_seed(42)
    # generate_contrasted("outputs/FCAPlusOther_mvtec_2023-07-12_11-08-33/logs/alpha_tiff",
    #                     "contrast", dataset_class=MVTecDataset, iterations=2000)
    # generate_contrasted("outputs/FCAPlusOther_MTD_2023-07-11_10-43-19/logs/alpha_tiff",
    #                     "contrast_bin03", dataset_class=MTDDataset, iterations=6000, significant_tau=0.002)
    # generate_contrasted("outputs/FCAPlusOther_coffee_2023-07-11_15-30-18/logs/alpha_tiff",
    #                     "contrast_bin03", dataset_class=CoffeeDataset, iterations=4000, significant_tau=0.002)
