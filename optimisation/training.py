import time
from tqdm import tqdm
import torch
from torchnet.meter import AverageValueMeter
from utils.metrics.psnr import PSNR
from utils.metrics.ssim import SSIM


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
            # Send inputs to correct device
            noisy = noisy.cuda() if args.cuda else noisy
            clean = clean.cuda() if args.cuda else clean

            iso = torch.Tensor(iso)
            iso = iso.cuda() if args.cuda else iso

            # Clear past gradients
            optimizer.zero_grad()

            # Denoise the image and calculate the loss wrt target clean image
            denoised = model(noisy, iso)
            loss = criterion(denoised, clean)

            # Calculate gradients and update weights
            loss.backward()
            optimizer.step()

            # Update meters
            loss_meter.add(loss.item())
            batch_time_meter.add(time.time() - end)
            end = time.time()

            # Update progress bar
            pbar.set_postfix(loss=loss_meter.mean)
            pbar.update()

            # Write the results to tensorboard
            summary_writer.add_scalar('Train/Loss', loss, (epoch * steps) + i)

    average_loss = loss_meter.mean
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
                # Send inputs to correct device
                noisy = noisy.cuda() if args.cuda else noisy
                clean = clean.cuda() if args.cuda else clean

                iso = torch.Tensor(iso)
                iso = iso.cuda() if args.cuda else iso

                # Denoise the image and calculate the loss wrt target clean image
                denoised = model(noisy, iso)
                loss = criterion(denoised, clean)

                # Update meters
                loss_meter.add(loss.item())
                batch_time_meter.add(time.time() - end)
                end = time.time()

                # Update progress bar
                pbar.set_postfix(loss=loss_meter.mean)
                pbar.update()

    average_loss = loss_meter.mean
    # Write average loss to tensorboard
    summary_writer.add_scalar('Test/Loss', average_loss, training_iters)

    print("===> Average total loss: {:4f}".format(average_loss))
    print("===> Average batch time: {:.4f}".format(batch_time_meter.mean))

    return average_loss


def evaluate_psnr_ssim(args, model, data_loader):
    # Average meters
    batch_time_meter = AverageValueMeter()
    psnr_meter = AverageValueMeter()
    ssim_meter = AverageValueMeter()

    psnr_calculator = PSNR(data_range=1)
    ssim_calculator = SSIM(data_range=1, channels=3)

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        steps = len(data_loader)
        # Start progress bar. Maximum value = number of batches.
        with tqdm(total=steps) as pbar:
            # Iterate through the validation batch samples
            for i, sample in enumerate(train_loader):
                noisy = sample['noisy']
                clean = sample['clean']
                iso = sample['iso']
                # Send inputs to correct device
                noisy = noisy.cuda() if args.cuda else noisy
                clean = clean.cuda() if args.cuda else clean

                iso = torch.Tensor(iso)
                iso = iso.cuda() if args.cuda else iso

                # Denoise the image and calculate the loss wrt target clean image
                denoised = model(noisy, iso)
                psnr = psnr_calculator(denoised, clean)
                ssim = ssim_calculator(denoised, clean)

                # Update meters
                psnr_meter.add(psnr.item())
                ssim_meter.add(ssim.item())

                batch_time_meter.add(time.time() - end)
                end = time.time()

                # Update progress bar
                pbar.set_postfix(psnr=psnr_meter.mean)
                pbar.set_postfix(ssim=ssim_meter.mean)
                pbar.update()

    average_psnr = psnr_meter.mean
    average_ssim = ssim_meter.mean
    # Write average loss to tensorboard
    print("===> Average PSNR score: {:4f}".format(average_psnr))
    print("===> Average SSIM score: {:4f}".format(average_ssim))
    print("===> Average batch time: {:.4f}".format(batch_time_meter.mean))
    # TODO: Save results to a csv/text file

    return average_psnr, average_ssim
