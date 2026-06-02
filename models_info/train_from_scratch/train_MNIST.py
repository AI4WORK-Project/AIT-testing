'''Train MNIST with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from models import *
from utils import progress_bar, Logger

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', type=str, default='lenet', help='The model to use for training and testing.')
parser.add_argument('--datapath', type=str, default='/datasets/', help='The saved dataset path.')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

start_time = int(round(time.time() * 1000))
timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(start_time / 1000))
saved_log_name = f'./logs/{args.model}-MNIST-{timestamp}.log'
log = Logger(saved_log_name, level='debug')
log.logger.debug(f"[=== Model: {args.model}, Dataset: MNIST]")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(32),  # Upscale from 28x28 to 32x32
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

trainset = torchvision.datasets.MNIST(
    root=args.datapath, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(
    root=args.datapath, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = tuple(str(i) for i in range(10))

# Model
print('==> Building model..')
if args.model == 'vgg16':
    net = VGG('VGG16')
elif args.model == 'resnet18':
    net = ResNet18()
elif args.model == 'lenet':
    net = LeNet()
# Add more model conditions as needed

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def save_model(model, model_name):
    torch.save(model.state_dict(), model_name + '.pt')
    torch.save(model, model_name + '_whole.pth')
    print("Model state saved as", model_name + '.pt')
    print("Whole model saved as", model_name + '_whole.pth')


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded from", model_path)
    return model


def train(epoch):
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


def test(epoch, model_name):
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

    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

        save_model(net, model_name + '_MNIST')
        load_model(net, model_name + '_MNIST.pt')
        log.logger.info("Epoch: {}, Acc: {}".format(epoch, acc))


for epoch in range(start_epoch, start_epoch + 20):  # Fewer epochs for MNIST
    train(epoch)
    test(epoch, model_name=args.model)
    scheduler.step()
