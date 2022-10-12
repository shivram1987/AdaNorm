'''Train CIFAR100 with PyTorch.'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from utils import progress_bar
from diffGrad import diffGrad
from Radam import Radam
from AdaBelief import AdaBelief
from AdamNorm import AdamNorm
from diffGradNorm import diffGradNorm
from RadamNorm import RadamNorm
from AdaBeliefNorm import AdaBeliefNorm


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
#parser.add_argument('--lr', default=0.0001, type=float, help='learning rate'); lr1 = '0001'
parser.add_argument('--lr', default=0.001, type=float, help='learning rate'); lr1 = '001'
#parser.add_argument('--lr', default=0.01, type=float, help='learning rate'); lr1 = '01'
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# bs = 32 #set batch size
bs = 64 #set batch size
# bs = 128 #set batch size
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = VGG('VGG16'); net1 = 'vgg16'
#net = ResNet18(); net1 = 'ResNet18'
#net = ResNet50(); net1 = 'ResNet50'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(), lr=args.lr); optimizer1 = 'Adam'
#optimizer = diffGrad(net.parameters(), lr=args.lr); optimizer1 = 'diffGrad'
#optimizer = Radam(net.parameters(), lr=args.lr); optimizer1 = 'Radam'
#optimizer = AdaBelief(net.parameters(), lr=args.lr); optimizer1 = 'AdaBelief'
optimizer = AdamNorm(net.parameters(), lr=args.lr); optimizer1 = 'AdamNorm'
#optimizer = diffGradNorm(net.parameters(), lr=args.lr); optimizer1 = 'diffGradNorm'
#optimizer = RadamNorm(net.parameters(), lr=args.lr); optimizer1 = 'RadamNorm'
#optimizer = AdaBeliefNorm(net.parameters(), lr=args.lr); optimizer1 = 'AdaBeliefNorm'

if args.resume:
    path_to_file = './CheckpointsResults/CIFAR100_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'_intermediate.t7'
    # if os.path.exists(path_to_file):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('CheckpointsResults'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(path_to_file)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if not os.path.isdir('CheckpointsResults'):
    os.mkdir('CheckpointsResults')

# Training
def train(epoch):
    f = open('./CheckpointsResults/CIFAR100_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'.txt', 'a')
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        if (batch_idx + 1) == len(trainloader):
            f.write('Train | Epoch: %d | Loss: %.3f | Acc: %.3f\n'
                % (epoch, train_loss / (batch_idx + 1), 100. * correct / total))
    f.close()

def test(epoch):
    f = open('./CheckpointsResults/CIFAR100_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'.txt', 'a')
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            if (batch_idx + 1) == len(testloader):
                f.write('Test | Epoch: %d | Loss: %.3f | Acc: %.3f\n'
                    % (epoch, test_loss / (batch_idx + 1), 100. * correct / total))
    f.close()
    # Save checkpoint.
    acc = 100. * correct / total
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('CheckpointsResults'):
        os.mkdir('CheckpointsResults')
    torch.save(state, './CheckpointsResults/CIFAR100_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'_intermediate.t7')
        
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('CheckpointsResults'):
            os.mkdir('CheckpointsResults')
        torch.save(state, './CheckpointsResults/CIFAR100_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'.t7')
        best_acc = acc

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1, last_epoch=-1)

for epoch in range(start_epoch, 100):
    scheduler.step()
    train(epoch)
    test(epoch)

f = open('./CheckpointsResults/CIFAR100_B'+str(bs)+'_LR'+lr1+'_'+net1+'_'+optimizer1+'.txt', 'a')
f.write('Best Accuracy:  %.3f\n'
    % (best_acc))
f.close()

print("Best Accuracy: ", best_acc)

