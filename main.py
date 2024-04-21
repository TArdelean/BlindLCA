import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn import cluster, mixture
import sklearn.metrics
import json

import op_utils
from data import CustomDataLoader
from op_utils import save_clusters, compute_cluster_f1


def compute_features(dataset_provider, method, cfg):
    for dataset in dataset_provider.get_instances():
        print(dataset.object_name, len(dataset))
        data_loader = CustomDataLoader(dataset_provider.feature_provider, dataset, batch_size=4)
        alphas = []
        bags = []
        for alpha, bag_f in method.call(data_loader):
            alphas.append(alpha)
            bags.append(bag_f)
        features = torch.stack(bags).detach().cpu().numpy()  # N x D
        np.save(f"bf_{dataset.object_name}.npy", features)
        if cfg.save_alpha_unit:
            op_utils.save_alpha_unit(alphas, dataset)
        if cfg.save_alpha_tiff:
            op_utils.save_alpha_tiff(alphas, dataset)


def clustering(dataset_provider, clustering_type):
    metrics = {}

    for dataset in dataset_provider.get_instances():
        features = np.load(f"bf_{dataset.object_name}.npy")
        print(dataset.object_name, features.shape)

        if clustering_type == "ward":
            cm = cluster.AgglomerativeClustering(n_clusters=dataset.label_ids.max() + 1, linkage='ward')
        elif clustering_type == "kmeans":
            cm = cluster.KMeans(n_clusters=dataset.label_ids.max() + 1, random_state=42)
        elif clustering_type == "spectral":
            cm = cluster.SpectralClustering(n_clusters=dataset.label_ids.max() + 1, random_state=42)
        elif clustering_type == "gaussian":
            cm = mixture.GaussianMixture(n_components=dataset.label_ids.max() + 1,
                                         covariance_type="full", random_state=42)
        else:
            raise Exception(f"Invalid clustering type {clustering_type}")
        cm.fit(features)
        if clustering_type == "gaussian":
            labels_ = cm.predict(features)
        else:
            labels_ = cm.labels_

        nmi = sklearn.metrics.normalized_mutual_info_score(dataset.label_ids, labels_)
        ari = sklearn.metrics.adjusted_rand_score(dataset.label_ids, labels_)
        print(dataset.label_names)
        f1 = compute_cluster_f1(dataset.label_ids, labels_)
        print("\tNMI: ", nmi)
        print("\tARI: ", ari)
        print("\tF1: ", f1)

        metrics[dataset.object_name] = dict(nmi=nmi, ari=ari, f1=f1)
        save_clusters(dataset, labels_)

    nmi_mean = np.mean([m['nmi'] for object_name, m in metrics.items()])
    ari_mean = np.mean([m['ari'] for object_name, m in metrics.items()])
    f1_mean = np.mean([m['f1'] for object_name, m in metrics.items()])

    print("Average NMI: ", nmi_mean)
    print("Average ARI: ", ari_mean)
    print("Average F1: ", f1_mean)
    metrics['avg_nmi'] = nmi_mean
    metrics['avg_ari'] = ari_mean
    metrics['avg_f1'] = f1_mean
    json.dump(metrics, open("metrics.json", "w"), indent=4)


@hydra.main(version_base=None, config_path="conf", config_name="base")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    dataset_provider = instantiate(cfg.dataset.provider)
    method = instantiate(cfg.method)
    compute_features(dataset_provider, method, cfg)
    clustering(dataset_provider, cfg.clustering)


if __name__ == "__main__":
    my_app()
