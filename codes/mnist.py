'''
Author: Kai Zhang
Date: 2022-03-19 15:12:04
LastEditors: Kai Zhang
LastEditTime: 2022-03-19 16:06:18
Description: demo of mnist classification 
'''
from datetime import datetime
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from tqdm import tqdm

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def train(gpu, args):
    model = ConvNet()
    model.cuda(gpu)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=None)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(gpu)
            labels = labels.to(gpu)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                   loss.item()))
    print("Training complete in: " + str(datetime.now() - start))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpuid', default=0, type=int,
                        help="which gpu to use")
    parser.add_argument('-e', '--epochs', default=1, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batchsize', default=4, type=int, 
                        metavar='N',
                        help='number of batchsize')         

    args = parser.parse_args()
    train(args.gpuid, args)

if __name__ == '__main__':
    main()