import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from data_sets import TinyImageNet
from model import AlexNet


train_data_path = './data/tiny-imagenet-200/train'
test_data_path = './data/tiny-imagenet-200/val'

# train_transform
train_transform = transforms.Compose([transforms.ToTensor()])
# test transform
test_transform = transforms.Compose([transforms.ToTensor()])

#dataset preparation
train_datasets = TinyImageNet(path=train_data_path, transform=train_transform)
test_datasets = TinyImageNet(path=test_data_path, transform=test_transform)

## Hyperparameters
EPOCHS = 10
LR = 3e-4
BATCH_SIZE = 2
DECAY = 0.0001

best_accuracy = 0.0

# DataLoader for train and test datasets
train_dataloader = DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_datasets, batch_size=BATCH_SIZE)

#Define model
model = AlexNet(n_ch=3, out_ch=10, drop_out=0.5)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=LR, weight_decay=DECAY)


def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            #run the model on test sets 
            outputs = model(images)

            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            accuracy += (predicted==labels).sum().item()

    # compute the accuracy over all the test images
    accuracy = (100 * accuracy/total)
    
    return accuracy

def train():
    for epoch in range(EPOCHS):
        running_loss = 0
        for i, (images, labels) in enumerate(train_dataloader):
            
            optim.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optim.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0
         # Compute and print the average accuracy for this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
            
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            # saveModel()
            best_accuracy = accuracy

if __name__ == "__main__":
    train()