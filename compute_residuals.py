import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import op_utils
from data import MVTecDataset, IndicesDataset, LeavesDataset, MTDDataset
from data.feature_provider import WideFeatures


class VariationalEncoder(nn.Module):
    def __init__(self, latent=32, nf=128, in_dim=512, kernel_size=1):
        super(VariationalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, nf, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(nf, nf, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.to_mu = nn.Conv2d(nf, latent, kernel_size=1)
        self.to_std = nn.Conv2d(nf, latent, kernel_size=1)

        self.kl = 0

    def forward(self, features):
        out = self.encoder(features)
        mu = self.to_mu(out)
        sigma = torch.exp(self.to_std(out))

        z = mu + sigma * torch.randn(mu.shape, device=features.device)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).mean()

        return z

    def inference(self, features):
        out = self.encoder(features)
        mu = self.to_mu(out)
        return mu


class Decoder(nn.Module):
    def __init__(self, latent=32, nf=128, in_dim=512, kernel_size=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(latent, nf, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(nf, nf, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(nf, in_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class VAE(nn.Module):
    def __init__(self, latent=128, nf=512, in_dim=512, kernel_size=1):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(latent, nf, in_dim, kernel_size=kernel_size)
        self.decoder = Decoder(latent, nf, in_dim, kernel_size=kernel_size)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    def inference(self, x):
        return self.decoder(self.encoder.inference(x))


def generate_vae_weights(tag, resize=512, cache_dir='cache', dataset_class=MVTecDataset, lambda_kl=0.001,
                         iterations=None, epochs=None, device=torch.device('cuda:0')):
    for object_name in dataset_class.objects:
        print(object_name)
        dataset = dataset_class(object_name=object_name, exclude_combined=True, resize=resize)
        fp = WideFeatures(device=device, resize=resize, save_in_memory=False, save_on_disk=True)
        fp.init(dataset)
        if iterations is not None:
            epochs = (iterations - 1) // len(dataset) + 1

        vae = VAE().to(device)
        optimizer = torch.optim.AdamW(vae.parameters(), lr=0.0001, weight_decay=0.1)
        loss_fn = nn.functional.mse_loss
        for _ in tqdm(range(epochs)):
            indices = IndicesDataset(torch.arange(0, len(dataset), device=device))
            data_loader = DataLoader(indices, batch_size=8, shuffle=True)
            for chunk in data_loader:
                features = fp.get(chunk)
                features = op_utils.scale_features(features)

                f_hat = vae(features)
                mse = loss_fn(f_hat, features)
                loss = mse + vae.encoder.kl * lambda_kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        path = Path(cache_dir) / dataset.data_root.name / tag
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(vae.state_dict(), str(path / f"{dataset.object_name}.pt"))


def generate_vae_diffs(tag, cache_dir='cache', dataset_class=MVTecDataset, resize=512, device=torch.device('cuda:0')):
    for object_name in dataset_class.objects:
        print(object_name)
        dataset = dataset_class(object_name=object_name, exclude_combined=False, resize=resize)
        path = Path(cache_dir) / dataset.data_root.name / tag

        fp = WideFeatures(device=device, resize=resize, save_in_memory=False, save_on_disk=False)
        fp.init(dataset)

        vae = VAE().to(device)
        vae.load_state_dict(torch.load(str(path / f"{dataset.object_name}.pt")))

        for i, img_path in enumerate(fp.img_paths):
            with torch.no_grad():
                features = fp.get([i])
                features = op_utils.scale_features(features)

                f_hat = vae.inference(features)
                residual = (features - f_hat)
            obj_name, class_name, img_name = dataset.path_tokens(img_path)
            (path / obj_name).mkdir(parents=True, exist_ok=True)
            file_path = path / object_name / f"{class_name}_{img_name.split('.')[0]}.pt"
            torch.save(residual[0], file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train VAE and compute residual features')
    parser.add_argument('-d', '--dataset', choices=['mvtec', 'leaves', 'mtd'])
    args = parser.parse_args()

    dataset_classes = {
        'mvtec': MVTecDataset,
        'leaves': LeavesDataset,
        'mtd': MTDDataset
    }
    dataset_class = dataset_classes[args.dataset]

    op_utils.set_seed(42)
    generate_vae_weights("VAE", resize=512, dataset_class=dataset_class, iterations=10000)
    generate_vae_diffs("VAE", dataset_class=dataset_class)
