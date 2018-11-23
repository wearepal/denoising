import time
from tqdm import tqdm
import torch
from torchnet.meter import AverageValueMeter
import torchvision.utils as vutils

from utils.metrics.psnr import PSNR
from utils.metrics.ssim import SSIM
from optimisation.loss import VGGLoss


def train(args, train_loader, model, criterion, optimizer, epoch, summary_writer):
    # Meters to log batch time and loss
    batch_time_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    steps = len(train_loader)
    # Start progress bar. Maximum value = number of batches.
    with tqdm(total=steps) as pbar:
        # Iterate through the training batch samples
        for i, sample in enumerate(train_loader):
            noisy = sample['noisy']
            clean = sample['clean']
            iso = sample['iso']
            class_labels = sample['class'].squeeze(-1)

            # Send inputs to correct device
            noisy = noisy.cuda() if args.cuda else noisy
            clean = clean.cuda() if args.cuda else clean
            iso = iso.cuda() if args.cuda else iso
            class_labels = class_labels.cuda() if args.cuda else class_labels

            # Clear past gradients
            optimizer.zero_grad()

            # Denoise the image and calculate the loss wrt target clean image
            denoised = model(noisy, iso, class_labels)
            loss = criterion(denoised, clean)

            # Calculate gradients and update weights
            loss.backward()
            optimizer.step()

            # Update meters
            loss_meter.add(loss.item())
            batch_time_meter.add(time.time() - end)
            end = time.time()

            # Write image samples to tensorboard
            if i == 0 and summary_writer is not None:
                if args.train_batch_size >= args.num_samples_to_log:
                    log_images(noisy, denoised, clean, summary_writer,
                               args.num_samples_to_log, (epoch * steps) + i, 'Train')

            # Update progress bar
            pbar.set_postfix(loss=loss_meter.mean)
            pbar.update()

            # Write the results to tensorboard
            if summary_writer is not None:
                summary_writer.add_scalar('Train/Loss', loss, (epoch * steps) + i)

    average_loss = loss_meter.mean
    print("===> Average total loss: {:4f}".format(average_loss))
    print("===> Average batch time: {:.4f}".format(batch_time_meter.mean))

    return average_loss


def train_gan(args, train_loader, generator, discriminator, content_criterion,
              adv_criterion, gen_optimizer, disc_optimizer, epoch, summary_writer):
    # Meters to log batch time and loss
    batch_time_meter = AverageValueMeter()
    total_loss_meter = AverageValueMeter()
    content_loss_meter = AverageValueMeter()
    adv_loss_meter = AverageValueMeter()

    end = time.time()
    steps = len(train_loader)
    # Start progress bar. Maximum value = number of batches.
    with tqdm(total=steps) as pbar:
        # Iterate through the training batch samples
        for i, sample in enumerate(train_loader):
            noisy = sample['noisy']
            clean = sample['clean']
            iso = sample['iso']
            class_labels = sample['class'].squeeze(-1)

            # Send inputs to correct device
            noisy = noisy.cuda() if args.cuda else noisy
            clean = clean.cuda() if args.cuda else clean
            iso = iso.cuda() if args.cuda else iso
            class_labels = class_labels.cuda() if args.cuda else class_labels
            # freeze generator's gradients; enable discriminator training

            discriminator.train()

            denoised = generator(noisy, iso, class_labels)
            # =========================
            # Train the discriminator
            # =========================
            for _ in range(args.disc_iters):
                # Clear past gradients
                gen_optimizer.zero_grad()
                disc_optimizer.zero_grad()

                disc_loss = adv_criterion(denoised, clean, discriminator)

                disc_loss.backward()
                disc_optimizer.step()

            # ====================
            # Train the generator
            # ====================
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            generator_content_loss = content_criterion(denoised, clean)
            generator_adversarial_loss = -discriminator(denoised).mean()  # applies only to wasserstein and hinge loss
            generator_total_loss = generator_content_loss + args.adv_weight * generator_adversarial_loss

            # Calculate gradients and update weights
            generator_total_loss.backward()
            gen_optimizer.step()

            # Update meters
            total_loss_meter.add(generator_total_loss.item())
            content_loss_meter.add(generator_content_loss.item())
            adv_loss_meter.add(generator_adversarial_loss.item())
            batch_time_meter.add(time.time() - end)
            end = time.time()

            # Write image samples to tensorboard
            if i == 0:
                if args.train_batch_size >= args.num_samples_to_log:
                    log_images(noisy, denoised, clean, summary_writer,
                               args.num_samples_to_log, (epoch * steps) + i, 'Train')

            # Update progress bar
            pbar.set_postfix(total_loss=total_loss_meter.mean,
                             content_loss=content_loss_meter.mean,
                             adv_loss=adv_loss_meter.mean)
            pbar.update()

            # Write the results to tensorboard
            training_iters = (epoch * steps) + i
            summary_writer.add_scalar('Train/Content_loss', generator_content_loss, training_iters)
            summary_writer.add_scalar('Train/Adversarial_loss', generator_adversarial_loss, training_iters)
            summary_writer.add_scalar('Train/Total_generator_loss', generator_total_loss, training_iters)

    average_loss = total_loss_meter.mean
    print("===> Average total loss: {:4f}".format(average_loss))
    print("===> Average batch time: {:.4f}".format(batch_time_meter.mean))

    return average_loss


def validate(args, val_loader, model, criterion, training_iters, summary_writer):
    """
    Args:
        args: Parsed arguments
        val_loader: Dataloader for validation data
        model: Denoising model
        criterion: Loss function
        training_iters: Number of training iterations elapsed
        summary_writer: Tensorboard summary writer

    Returns:
        Average loss on validation samples
    """
    # Average meters
    batch_time_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        steps = len(val_loader)
        # Start progress bar. Maximum value = number of batches.
        with tqdm(total=steps) as pbar:
            # Iterate through the validation batch samples
            for i, sample in enumerate(val_loader):
                noisy = sample['noisy']
                clean = sample['clean']
                iso = sample['iso']
                class_labels = sample['class'].squeeze(-1)

                # Send inputs to correct device
                noisy = noisy.cuda() if args.cuda else noisy
                clean = clean.cuda() if args.cuda else clean
                iso = iso.cuda() if args.cuda else iso
                class_labels = class_labels.cuda() if args.cuda else class_labels

                # Denoise the image and calculate the loss wrt target clean image
                denoised = model(noisy, iso, class_labels)
                loss = criterion(denoised, clean)

                # Update meters
                loss_meter.add(loss.item())
                batch_time_meter.add(time.time() - end)
                end = time.time()

                # Write image samples to tensorboard
                if i == 0:
                    if args.test_batch_size >= args.num_samples_to_log:
                        log_images(noisy, denoised, clean, summary_writer,
                                   args.num_samples_to_log, training_iters, 'Val')

                # Update progress bar
                pbar.set_postfix(loss=loss_meter.mean)
                pbar.update()

    average_loss = loss_meter.mean
    # Write average loss to tensorboard
    summary_writer.add_scalar('Test/Loss', average_loss, training_iters)

    print("===> Average total loss: {:4f}".format(average_loss))
    print("===> Average batch time: {:.4f}".format(batch_time_meter.mean))

    return average_loss


def log_images(noisy_image, denoised_image, clean_image,
               summary_writer, n_samples, training_iters, prefix):
    summary_writer.add_image(
        str(prefix) + '/denoised_images', vutils.make_grid(denoised_image.data[:n_samples], normalize=True,
                                                           scale_each=True), training_iters)
    summary_writer.add_image(
        str(prefix) + '/clean_images', vutils.make_grid(clean_image.data[:n_samples], normalize=True,
                                                        scale_each=True), training_iters)
    summary_writer.add_image(
        str(prefix) + '/noisy_images', vutils.make_grid(noisy_image.data[:n_samples], normalize=True,
                                                        scale_each=True), training_iters)


def evaluate_psnr_and_vgg_loss(args, model, data_loader):
    # Average meters
    batch_time_meter = AverageValueMeter()
    psnr_meter = AverageValueMeter()
    ssim_meter = AverageValueMeter()
    vgg_loss_meter = AverageValueMeter()

    psnr_calculator = PSNR(data_range=1)
    ssim_calculator = SSIM(data_range=1, channels=args.cnn_in_channels)
    vgg_loss_calculator = VGGLoss(args)

    if args.cuda:
        psnr_calculator.cuda()
        ssim_calculator.cuda()
        vgg_loss_calculator.cuda()

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        steps = len(data_loader)
        # Start progress bar. Maximum value = number of batches.
        with tqdm(total=steps) as pbar:
            # Iterate through the validation batch samples
            for i, sample in enumerate(data_loader):
                noisy = sample['noisy']
                clean = sample['clean']
                iso = sample['iso']
                class_labels = sample['class'].squeeze(-1)

                # Send inputs to correct device
                noisy = noisy.cuda() if args.cuda else noisy
                clean = clean.cuda() if args.cuda else clean
                iso = iso.cuda() if args.cuda else iso
                class_labels = class_labels.cuda() if args.cuda else class_labels

                # Denoise the image and calculate the loss wrt target clean image
                denoised = model(noisy, iso, class_labels)
                psnr = psnr_calculator(denoised, clean).mean()
                ssim = ssim_calculator(denoised, clean).mean()
                vgg_loss = vgg_loss_calculator(denoised, clean)

                # Update meters
                psnr_meter.add(psnr.item())
                ssim_meter.add(ssim.item())
                vgg_loss_meter.add(vgg_loss.item())

                batch_time_meter.add(time.time() - end)
                end = time.time()

                # Update progress bar
                pbar.set_postfix(psnr=psnr_meter.mean)
                pbar.set_postfix(ssim=ssim_meter.mean)
                pbar.set_postfix(ssim=vgg_loss_meter.mean)
                pbar.update()

    average_psnr = psnr_meter.mean
    average_ssim = ssim_meter.mean
    average_vgg_loss = vgg_loss_meter.mean
    # Write average loss to tensorboard
    print("===> Average PSNR score: {:4f}".format(average_psnr))
    print("===> Average SSIM score: {:.4f}".format(average_ssim))
    print("===> Average VGG loss: {:4f}".format(average_vgg_loss))
    print("===> Average batch time: {:.4f}".format(batch_time_meter.mean))
    # TODO: Save results to a csv/text file

    return average_psnr, average_ssim, average_vgg_loss
