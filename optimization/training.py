import time
from tqdm import tqdm
import torch
from torchnet.meter import AverageValueMeter


def train(args, train_loader, model, criterion, optimizer, epoch, summary_writer):
    # Meters to log batch time and loss
    batch_time = AverageValueMeter()
    loss = AverageValueMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    steps = len(train_loader)
    # Start progress bar. Maximum value = number of batches.
    with tqdm(total=steps) as pbar:
        # Iterate through the training batch samples
        for i, (noisy, clean, iso) in enumerate(train_loader):
            # Send inputs to correct device
            noisy = noisy.cuda() if args.cuda else noisy
            clean = clean.cuda() if args.cuda else clean
            # ISO needs to be a 3d tensor to be passed to Gated Convolutions
            iso = torch.FloatTensor(iso.values).view(noisy.size(0), -1, 1)
            if args.cuda:
                iso.cuda()

            # Clear past gradients
            optimizer.zero_grad()

            # Denoise the image and calculate the loss wrt target clean image
            denoised = model(noisy, iso)
            loss = criterion(denoised, clean)

            # Calculate gradients and update weights
            loss.backward()
            optimizer.step()

            # Update meters
            loss.add(loss.item())
            batch_time.add(time.time() - end)
            end = time.time()

            # Update progress bar
            pbar.set_postfix(loss=loss.value()[0])
            pbar.update()

            # Write the results to tensorboard
            summary_writer.add_scalar('Train/Loss', loss, (epoch * steps) + i)

    print("===> Average total loss: {:4f}".format(loss.value()[0]))
    print("===> Average batch time: {:.4f}".format(batch_time.value()[0]))

    return loss.value()[0]


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
    batch_time = AverageValueMeter()
    loss = AverageValueMeter()

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        steps = len(val_loader)
        # Start progress bar. Maximum value = number of batches.
        with tqdm(total=steps) as pbar:
            # Iterate through the validation batch samples
            for i, (noisy, clean, iso) in enumerate(val_loader):
                # Send inputs to correct device
                noisy = noisy.cuda() if args.cuda else noisy
                clean = clean.cuda() if args.cuda else clean
                # ISO needs to be a 3d tensor to be passed to Gated Convolutions
                iso = torch.FloatTensor(iso.values).view(noisy.size(0), -1, 1)
                if args.cuda:
                    iso.cuda()

                # Denoise the image and calculate the loss wrt target clean image
                denoised = model(noisy, iso)
                loss = criterion(denoised, clean)

                # Calculate gradients and update weights

                # Update meters
                loss.add(loss.item())
                batch_time.add(time.time() - end)
                end = time.time()

                # Update progress bar
                pbar.set_postfix(loss=loss.value()[0])
                pbar.update()

    average_loss = loss.value()[0]
    # Write average loss to tensorboard
    summary_writer.add_scalar('Test/Loss', average_loss, training_iters)

    print("===> Average total loss: {:4f}".format(average_loss))
    print("===> Average batch time: {:.4f}".format(batch_time.value()[0]))

    return average_loss
