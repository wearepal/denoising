"""Entry point for GAN models"""
from sys import argv
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from optimisation.testing import test
from optimisation.training import train, train_gan, validate, evaluate
from optimisation import loss
from utils import TransformedHuaweiDataset, transform_sample, parse_arguments
from utils.functions import apply_spectral_norm
import models


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda:
        # gpu device number
        torch.cuda.set_device(args.gpu_num)

    if args.test_data_dir:
        test(args, transform_sample)
        return

    # Create results path
    if args.save_dir:  # If specified
        save_path = Path(args.save_dir).resolve()
    else:
        save_path = Path().resolve().parent / "results" / args.generator / str(round(time.time()))
        save_path.parent.mkdir(exist_ok=True)
    save_path.mkdir()  # Will throw an exception if the path exists OR the parent path _doesn't_

    kwargs = {'pin_memory': True} if args.cuda else {}

    print('\nMODEL SETTINGS: \n', args.state_dict(), '\n')
    print("Random Seed: ", args.seed)

    # Save config
    torch.save(args, save_path / 'denoising.config')
    writer = SummaryWriter(save_path / 'summaries')

    # generator
    generator = getattr(models, args.generator)(args)
    apply_spectral_norm(generator)  # apply spectral normalization to all generator layers
    generator = generator.cuda() if args.cuda else generator

    # discriminator
    discriminator = getattr(models, args.discriminator)(args)
    apply_spectral_norm(discriminator)  # apply spectral normalization to all discriminator layers
    discriminator = discriminator.cuda() if args.cuda else discriminator

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args.gen_learning_rate,
                                     betas=(args.beta1, args.beta2))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.disc_learning_rate,
                                      betas=(args.beta1, args.beta2))

    criterion_constructor = getattr(loss, args.content_loss)
    content_criterion = criterion_constructor(args) if args.args_to_loss else criterion_constructor()
    content_criterion = content_criterion.cuda() if args.cuda else content_criterion

    adv_criterion = getattr(loss, args.adv_loss)()
    adv_criterion = adv_criterion.cuda() if args.cuda else adv_criterion

    dataset = TransformedHuaweiDataset(root_dir=args.data_dir, transform=transform_sample)
    train_dataset, val_dataset = dataset.random_split(test_ratio=args.test_split,
                                                      data_subset=args.data_subset)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                              shuffle=True, num_workers=args.workers, **kwargs)

    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size,
                            shuffle=False, num_workers=args.workers, **kwargs)

    best_loss = np.inf

    if args.resume:
        print('==> Loading checkpoint')
        checkpoint = torch.load(args.resume)
        print('==> Checkpoint loaded')
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            generator.load_state_dict(checkpoint['model'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])

    if args.evaluate:
        # Evaluate model using PSNR and SSIM metrics
        evaluate(args, generator, val_loader)
        return

    # pre-train generator
    for epoch in range(args.pretrain_epochs):
        print("===> Pre-training generator")
        train(args, train_loader, generator, content_criterion, gen_optimizer, epoch, None)

    for epoch in range(args.start_epoch, args.epochs):
        training_iters = (epoch + 1) * len(train_loader)

        # Train
        print("===> Training on Epoch %d" % epoch)
        train_gan(args, train_loader, generator, discriminator,
                  content_criterion, adv_criterion,
                  gen_optimizer, disc_optimizer,
                  epoch, writer)

        # Validate
        print("===> Validating on Epoch %d" % epoch)
        val_loss = validate(args, val_loader, generator, content_criterion, training_iters, writer)

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        # Save checkpoint
        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        checkpoint = {
            'epoch': epoch,
            'model': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'disc_optimizer': disc_optimizer.state_dict(),
            'best_loss': best_loss
        }
        save_checkpoint(checkpoint, model_filename, is_best, save_path)

    # Evaluate model using PSNR and SSIM metrics
    evaluate(args, generator, val_loader)


def save_checkpoint(checkpoint, filename, is_best, save_path):
    print("===> Saving checkpoint '{}'".format(filename))
    model_filename = save_path / filename
    best_filename = save_path / 'model_best.pth.tar'
    torch.save(checkpoint, model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)
    print("===> Saved checkpoint '{}'".format(model_filename))


if __name__ == '__main__':
    main(parse_arguments(argv[1] if len(argv) >= 2 else "run_configs/default_gan.ini"))
