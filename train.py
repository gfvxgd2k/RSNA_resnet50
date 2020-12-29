import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import f1_score, confusion_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float,default=0.01, help='')
parser.add_argument('--data',type=str, default='./', help='')
parser.add_argument('--bs', type=int,default=256, help='')
parser.add_argument('--num_workers', default=8, type=int,help='')
parser.add_argument('--gpu', default=0, type=int, help='')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')



def main():
    args = parser.parse_args()
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = torchvision.models.resnet50(pretrained=False, progress=True,)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    validation_ratio = 0.1
    random_seed = 10

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1, last_epoch=-1)

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_validation = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    traindir = os.path.join(args.data, 'train')
    
    trainset = datasets.ImageFolder(
    traindir, transform=transform_train)

    validset = datasets.ImageFolder(
    traindir, transform=transform_validation)

    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(validation_ratio * num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.bs, sampler=train_sampler, pin_memory=True, num_workers=args.num_workers)

    valid_loader = torch.utils.data.DataLoader(
    validset, batch_size=args.bs, sampler=valid_sampler, pin_memory=True, num_workers=args.num_workers)

    best_loss = 0
    for epoch in range(args.epochs):
        lr_scheduler.step()
        running_loss = 0.0
        loss = train(train_loader, model, criterion, optimizer, epoch, args, running_loss)
        valid(valid_loader,model,epoch,args)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_chekpoint({
            'epoch':epoch + 1,
            'state_dict':model.state_dict(),
            'best_loss':loss,
            'optimizer':optimizer.state_dict()
        },is_best)
    print('Finished Training')

def save_chekpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,'model_best.pth.tar')

def train(train_loader, model, criterion, optimizer, epoch, args, running_loss):
    model.train()
    loss =0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(args.gpu), labels.cuda(args.gpu)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        print('Epoch:[{}] [{}/{}]\t'
              'Loss: {:.4f}'.format(epoch, i, len(train_loader),loss.item()))
    return loss        

def valid(valid_loader,model,epoch,args):
    cpu = torch.device('cpu')
    correct = 0
    total = 0
    with torch.no_grad():
        sum_f1 = 0
        sum_cf = 0
        count = 1
        for _, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(args.gpu), labels.cuda(args.gpu)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            labels_cpu = labels.to(cpu)
            predicted_cpu = predicted.to(cpu)
            f1 = f1_score(labels_cpu,predicted_cpu)
            cf = confusion_matrix(labels_cpu,predicted_cpu)
            sum_f1 += f1
            sum_cf += cf
            count += 1
        print('Epoch:[{}]\t'
              'Accuracy: {:.4f}%   F1-score: {:.4f}\t'   
              'Confusion-matrix: {}'.format(epoch + 1,100 * correct / total, sum_f1/count, sum_cf/count))
    
if __name__ == '__main__':
    main()


