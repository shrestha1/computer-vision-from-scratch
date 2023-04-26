import os
import torch
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist
from model import VGG

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

def init_distributed():

    # Initialize the distributed backend which will
    # take care of synchronizing nodes/GPUs
    dist_url = "env://" #default

    # only works with torch.distributed.launch // torch run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

    # all .cuda() calls to work properly
    torch.cuda.set_device(local_rank)

    #sychronize before all the threads 
    dist.barrier()

def create_data_loader_cifar10():
    '''
    ref: from AI summer 
    '''

    transform = transforms.Compose(
        [
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)                                  
    train_sampler = DistributedSampler(dataset=trainset, shuffle=True)                                                  
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=train_sampler, num_workers=10, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_sampler =DistributedSampler(dataset=testset, shuffle=True)                                         
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, sampler=test_sampler, num_workers=10)

    return trainloader, testloader

def train(net, trainloader):

    print("start training....")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    epochs = 1
    num_of_batches = len(trainloader)

    for epoch in range(epochs):
        ##
        trainloader.sampler.set_epoch(epoch)
        for i, data in enumerate(trainloader, 0):

           # get the inputs; data is a list of [inputs, labels]
           inputs, labels = data
           images, labels = inputs.cuda(), labels.cuda()

           # zero the parameter gradients
           optimizer.zero_grad()

           # forward + backward + optimize
           outputs = net(images)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()

           # print statistics
           running_loss += loss.item()

        print(f'[Epoch {epoch + 1}/{epochs}] loss: {running_loss / num_of_batches:.3f}')

    print('Finished Training')

if __name__ == '__main__':
    net = VGG.vgg11()
    #chaning the last layer of network to adapt with CIFAR10 classes
    net.fc_layer[-1] = torch.nn.Linear(in_features=4096, out_features=10)
    
    trainloader, testloader = create_data_loader_cifar10()
    
    # train the model
    train(net, trainloader=trainloader)

# python -m torch.distributed.launch --nproc_per_node=4 train.py