import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import LeNet

# transform data
data_transform = transforms.Compose(
    [
        transforms.Resize(size = (32, 32)),
        transforms.ToTensor()
    ]
)

# Prepare train dataset from Mnist data
train_datasets = datasets.MNIST(
    root='data',
    train=True,
    transform=data_transform,
    download=True
)

# Prepare test dataset from Mnist data
test_datasets = datasets.MNIST(
    root='data',
    train=False,
    transform=data_transform,
    download=True
)

# HyperParameters 
EPOCHS = 10
LR = 3e-4
BATCH_SIZE = 32
DECAY = 0.0001

best_acc = 0.0

# Prepare DataLoader for train and test datasets
train_loader = DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True )
test_loader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE)

# Define model
model = LeNet(1, 10)

# Define loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = LR, weight_decay = DECAY)

# Function to save the model
def saveModel():
    path = "./lenetmodel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with test datasets
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            #run the model on the test set to predict labels
            outputs = model(images)

            #the label with highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100*accuracy/total)

    return accuracy

# function to train a model

def train():
    best_acc = 0.0

    for epoch in range(EPOCHS):
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            
            # nullify all previous stored gradient if any
            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0
        
        # compute and print average accuracy
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))

        # saving the model if the accuracy is best
        if accuracy > best_acc:
            saveModel()
            best_acc = accuracy


if __name__ == '__main__':
    train()

