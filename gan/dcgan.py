import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
from torchvision import transforms


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='InsPLAD', help='Name of dataset')
    parser.add_argument('--source_class', type=int, default=0, help='Source class label')
    parser.add_argument('--target_class', type=int, default=1, help='Target class label')
    parser.add_argument('--n_classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--nz', type=int, default=100, help='Size of latent vector')
    parser.add_argument('--ngf', type=int, default=64, help='Size of generator feature maps')
    parser.add_argument('--ndf', type=int, default=64, help='Size of discriminator feature maps')
    parser.add_argument('--save_path', type=str, default='model_weight', help='Path to save model')

    parser.add_argument('--seed', type=int, default=2021, help='Random seed')
    parser.add_argument('--gen_total', type=int, default=1024, help='Total number of images to generate')
    parser.add_argument('--gen_save_dir', type=str, default='InsPLAD_gan_images/train/corrosao', help='Directory to save generated images')

    args = parser.parse_args()
    return args


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def build_generator(nz, ngf, nc, n_classes):
    return nn.Sequential(
        nn.ConvTranspose2d(nz + n_classes, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False), nn.Tanh()
    )


def build_discriminator(ndf, nc, n_classes):
    return nn.Sequential(
        nn.Conv2d(nc + n_classes, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Flatten(), nn.Sigmoid()
    )


def to_categorical(y: torch.FloatTensor, lb, device):
    y_one_hot = lb.transform(y.cpu())
    floatTensor = torch.FloatTensor(y_one_hot)
    return floatTensor.to(device)


def generate_images(args, netG, lb, device):
    os.makedirs(args.gen_save_dir, exist_ok=True)
    netG.eval()

    total = args.gen_total
    label = args.target_class
    batch_size = args.batch_size
    steps = int(np.ceil(total / batch_size))

    for step in range(steps):
        noise_z1 = torch.randn(batch_size, args.nz, 1, 1).to(device)
        gen_labels = np.full((batch_size,), label)
        gen_labels_tensor = torch.tensor(gen_labels).view(-1, 1)
        target = to_categorical(gen_labels_tensor, lb, device).unsqueeze(2).unsqueeze(3).float()
        noise_z = torch.cat((noise_z1, target), dim=1)

        with torch.no_grad():
            fake_data = netG(noise_z.to(device))

        data = fake_data.detach().cpu().permute(0, 2, 3, 1).numpy()
        data = (data * 0.5 + 0.5).clip(0, 1)

        for i in range(batch_size):
            if step * batch_size + i >= total:
                break
            save_path = os.path.join(args.gen_save_dir, f'{step * batch_size + i}.jpg')
            plt.imsave(save_path, data[i])


def main():
    args = args_parser()
    seed_torch(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'InsPLAD':
        nc = 3
    real_label = 1.0
    fake_label = 0.0

    netG = build_generator(args.nz, args.ngf, nc, args.n_classes)
    netD = build_discriminator(args.ndf, nc, args.n_classes)
    netG.apply(weights_init)
    netD.apply(weights_init)
    netG.to(device)
    netD.to(device)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.ImageFolder(root='InsPLAD_Gan/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lb = LabelBinarizer()
    lb.fit(list(range(args.n_classes)))

    os.makedirs(args.save_path, exist_ok=True)

    for epoch in range(args.epochs):
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = torch.full_like(target, args.target_class)

            target1 = to_categorical(target, lb, device).unsqueeze(2).unsqueeze(3).float()
            target2 = target1.repeat(1, 1, data.size(2), data.size(3))
            data_input = torch.cat((data, target2), dim=1)
            label = torch.full((data.size(0), 1), real_label).to(device)

            netD.zero_grad()
            output = netD(data_input)
            loss_D1 = criterion(output, label)

            noise_z = torch.randn(data.size(0), args.nz, 1, 1).to(device)
            noise_z = torch.cat((noise_z, target1), dim=1)
            fake_data = netG(noise_z)
            fake_input = torch.cat((fake_data, target2), dim=1)
            label_fake = torch.full((data.size(0), 1), fake_label).to(device)

            output = netD(fake_input.detach())
            loss_D2 = criterion(output, label_fake)
            loss_D = loss_D1 + loss_D2
            loss_D.backward()
            optimizerD.step()

            netG.zero_grad()
            label_real = torch.full((data.size(0), 1), real_label).to(device)
            output = netD(fake_input)
            loss_G = criterion(output, label_real)
            loss_G.backward()
            optimizerG.step()

            if batch % 10 == 0:
                print(f'Epoch [{epoch}/{args.epochs}] Batch [{batch}] '
                      f'Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}')
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, f'GAN_{args.dataset}_{epoch+1}.pth')
    torch.save({'net_G': netG.state_dict(), 'net_D': netD.state_dict(), 'start_epoch': epoch}, save_path)


    checkpoint_path = os.path.join(args.save_path, f'GAN_{args.dataset}_{args.epochs}.pth')
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['net_G'])
    generate_images(args, netG, lb, device)


if __name__ == '__main__':
    main()
