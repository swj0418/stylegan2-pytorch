import os
import sys
import time
import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

import numpy as np


def generate(args, g_ema, device):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
           sample_z = torch.randn(args.sample, args.latent, device=device)

           sample, _ = g_ema([sample_z])
           
           utils.save_image(
            sample,
            f'sample/{str(i).zfill(6)}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

def generate_specified_samples(args, g_ema, device):
    try:
        os.mkdir('latent_sample_single_slot')
    except:
        pass

    with torch.no_grad():
        g_ema.eval()

        for i in range(1, 10):
            sample_z = torch.zeros(args.sample, args.latent, device=device)
            r_slot = np.random.randint(0, 512)
            v_slot = np.random.randint(low=1, high=100)
            sample_z[0][r_slot] = v_slot
            sample, _ = g_ema([sample_z])

            utils.save_image(
                sample,
                f'latent_sample_single_slot/{str(i).zfill(6)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


def generate_larger_sample(args, g_ema, device):
    try:
        os.mkdir('latent_sample_single_slot')
    except:
        pass

    with torch.no_grad():
        g_ema.eval()

        samples = []
        for i in range(100):
            sample_z = torch.zeros(args.sample, args.latent, device=device)

            sample_z[0][13] = np.random.rand(1)[0]

            # r_idx_1 = np.random.randint(0, 512)
            # r_idx_2 = np.random.randint(0, 512)
            # r_idx_3 = np.random.randint(0, 512)
            # r_idx_4 = np.random.randint(0, 512)
            # r_idx_5 = np.random.randint(0, 512)
            # r_idx_6 = np.random.randint(0, 512)
            sample_z[0][46] = np.random.random(1)[0] * 100
            # sample_z[0][r_idx_2] = np.random.random(1)[0]
            # sample_z[0][r_idx_3] = np.random.random(1)[0]
            # sample_z[0][r_idx_4] = np.random.random(1)[0]
            # sample_z[0][r_idx_5] = np.random.random(1)[0]
            # sample_z[0][r_idx_6] = np.random.random(1)[0]
            sample, _ = g_ema([sample_z])

            sample = sample.detach().cpu().numpy()
            sample = np.squeeze(sample, axis=0)
            sample = np.einsum('kli->lik', sample)
            if i % 10 == 0:
                if i is not 0:
                    samples.append(sampleb)
                sampleb = sample

            sampleb = np.concatenate((sampleb, sample), axis=1)

            # print(sampleb.shape)
        image = samples[0]
        for s in range(1, len(samples)):
            image = np.concatenate((image, samples[s]), axis=0)

        image = np.einsum('kli->ikl', image)
        image = np.expand_dims(image, axis=0)
        image = torch.Tensor(image)
        print(image.shape)

        utils.save_image(
            image,
            f'latent_sample_single_slot/{str(i).zfill(6)}.png',
            normalize=True,
            range=(-1, 1),
        )


def generate_larger_overlaping_sample(args, g_ema, device):
    try:
        os.mkdir('latent_sample_single_slot')
    except:
        pass

    with torch.no_grad():
        g_ema.eval()

        samples = []
        for i in range(100):
            sample_z = torch.zeros(args.sample, args.latent, device=device)

            sample_z[0][10] = np.random.rand(1)[0]

            r_idx_1 = np.random.randint(0, 512)
            # r_idx_2 = np.random.randint(0, 512)
            # r_idx_3 = np.random.randint(0, 512)
            # r_idx_4 = np.random.randint(0, 512)
            # r_idx_5 = np.random.randint(0, 512)
            # r_idx_6 = np.random.randint(0, 512)
            sample_z[0][55] = np.random.random(1)[0] * 1
            # sample_z[0][r_idx_2] = np.random.random(1)[0]
            # sample_z[0][r_idx_3] = np.random.random(1)[0]
            # sample_z[0][r_idx_4] = np.random.random(1)[0]
            # sample_z[0][r_idx_5] = np.random.random(1)[0]
            # sample_z[0][r_idx_6] = np.random.random(1)[0]

            sample, _ = g_ema([sample_z])

            sample = sample.detach().cpu().numpy()
            sample = np.squeeze(sample, axis=0)
            sample = np.einsum('kli->lik', sample)
            slice_in = 156
            sample = sample[slice_in:-slice_in, slice_in:-slice_in, :]
            if i % 10 == 0:
                if i is not 0:
                    samples.append(sampleb)
                sampleb = sample

            sampleb = np.concatenate((sampleb, sample), axis=1)

            # print(sampleb.shape)
        image = samples[0]
        for s in range(1, len(samples)):
            image = np.concatenate((image, samples[s]), axis=0)

        image = np.einsum('kli->ikl', image)
        image = np.expand_dims(image, axis=0)
        image = torch.Tensor(image)
        print(image.shape)

        utils.save_image(
            image,
            f'latent_sample_single_slot/overlapped_larger_sample_' + str(time.time()) + '.png',
            normalize=True,
            range=(-1, 1),
        )


def list_styles(args, g_ema, device):
    try:
        os.mkdir('style_samples')
    except:
        pass

    with torch.no_grad():
        g_ema.eval()

        for i in range(1, 512):
            print(i)
            torch.manual_seed(i)
            sample_z = torch.zeros(args.sample, args.latent, device=device)
            sample_z[0][i] = np.random.uniform(low=0.0, high=1.0)
            sample, _ = g_ema([sample_z])

            utils.save_image(
                sample,
                f'style_samples/style_{str(i).zfill(3)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)

    parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    args.ckpt = '/home/sangwon/Work/NeuralNetworks/StyleGan/weights/stylegan2-1024-mult2-r1-10-asphalt-large/060000.pt'
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint['g_ema'])

    # generate_larger_sample(args, g_ema, device)
    generate_larger_overlaping_sample(args, g_ema, device)
    # generate_specified_samples(args, g_ema, device)
    # list_styles(args, g_ema, device)
    generate(args, g_ema, device)
