import torch
import torch.nn as nn
from utils.multi_loaders import create_datasets
import os


def test_simple():
    # Hyper parameters
    num_epochs = 5
    learning_rate = 0.001
    max_train_samples = 20
    crops_per_image = 3
    ratio = 0.99
    batch_size = 4

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = create_datasets(os.path.expanduser("~/Downloads/huawei_ai"),
                                                train_ratio=ratio,
                                                max_train_samples=max_train_samples,
                                                crops_per_image=crops_per_image,
                                                batch_size=batch_size)

    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(8192, 1)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out

    model = ConvNet().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, d in enumerate(train_loader):
            images = d['noisy'].to(device)
            labels = d['iso'].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    total_loss = 0
    with torch.no_grad():
        for d in test_loader:
            images = d['noisy'].to(device)
            labels = d['iso'].to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels)

    print("total loss:", total_loss)
