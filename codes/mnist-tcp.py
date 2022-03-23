'''
Author: Kai Zhang
Date: 2022-03-19 16:05:57
LastEditors: Kai Zhang
LastEditTime: 2022-03-23 21:24:58
Description: Start DDP in tcp model
'''
from datetime import datetime
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.cuda.amp import GradScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpuid', default=0, type=int,
                        help="which gpu to use")
    parser.add_argument('-e', '--epochs', default=1, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N',
                        help='number of batchsize')
    ##################################################################################
    parser.add_argument('--init_method', default='tcp://localhost:18888',            #
                        help="init-method")                                          #
    parser.add_argument('-r', '--rank', default=0, type=int,                         #
                    help='rank of current process')                                  #
    parser.add_argument('--world_size', default=2, type=int,                         #
                        help="world size")                                           #
    parser.add_argument('--use_mix_precision', default=False,                        #
                        action='store_true', help="whether to use mix precision")    #
    ##################################################################################
    args = parser.parse_args()
    train(args.gpuid, args)

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


####################################    N11    ##################################
def evaluate(model, gpu, test_loader, rank):
    model.eval()
    size = torch.tensor(0.).to(gpu)
    correct = torch.tensor(0.).to(gpu)
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(gpu)
            labels = labels.to(gpu)
            # Forward pass
            outputs = model(images)
            size += images.shape[0]
            correct += (outputs.argmax(1) == labels).type(torch.float).sum()
    # 群体通信 reduce 操作 change to allreduce if Gloo
    dist.reduce(size, 0, op=dist.ReduceOp.SUM)
    # 群体通信 reduce 操作 change to allreduce if Gloo
    dist.reduce(correct, 0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Evaluate accuracy is {:.2f}'.format(correct / size))
 #################################################################################

def train(gpu, args):
    ########################################    N1    ####################################################################
    dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.world_size)    #
    ######################################################################################################################
    model = ConvNet()
    model.cuda(gpu)
    # Wrap the model
    #######################################    N2    ########################
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)                  #
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])    #
    scaler = GradScaler(enabled=args.use_mix_precision)                     #
    #########################################################################
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    ####################################    N3    #######################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)      #
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,                   #
                                               batch_size=args.batch_size,              #
                                               shuffle=False,                           #
                                               num_workers=0,                           #
                                               pin_memory=True,                         #
                                               sampler=train_sampler)                   #
    #####################################################################################

    ####################################    N9    ###################################
    test_dataset = torchvision.datasets.MNIST(root='./data',                        #
                                               train=False,                         #
                                               transform=transforms.ToTensor(),     #
                                               download=True)                       #
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)    #
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,                 #
                                               batch_size=args.batch_size,          #
                                               shuffle=False,                       #
                                               num_workers=0,                       #
                                               pin_memory=True,                     #
                                               sampler=test_sampler)                #
    #################################################################################
    start = datetime.now()
    total_step = len(train_loader) # The number changes to orignal_length // args.world_size
    for epoch in range(args.epochs):
        ################    N4    ################
        train_loader.sampler.set_epoch(epoch)    #
        ##########################################
        model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(gpu)
            labels = labels.to(gpu)
            # Forward pass
            ########################    N5    ################################
            with torch.cuda.amp.autocast(enabled=args.use_mix_precision):    #
                outputs = model(images)                                      #
                loss = criterion(outputs, labels)                            #
            ##################################################################
            # Backward and optimize
            optimizer.zero_grad()
            ##############    N6    ##########
            scaler.scale(loss).backward()    #
            scaler.step(optimizer)           #
            scaler.update()                  #
            ##################################
            ################    N7    ####################
            if (i + 1) % 100 == 0 and args.rank == 0:    #
            ##############################################
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                   loss.item()))
        evaluate(model, gpu, test_loader, args.rank)
    dist.destroy_process_group()
    ######    N8    #######
    if args.rank == 0:    #
    #######################
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()
