import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CNN

# train transform
train_transform = transforms.Compose([transforms.Resize(size = (32, 32)), transforms.ToTensor()])

train_datasets = datasets.MNIST(
    root='data',
    train=True,
    transform=train_transform,
    download= True
)

test_datasets = datasets.MNIST(
    root = 'data',
    train=False,
    transform= train_transform,
    download=True
)

# Hyperparameters
EPOCHS = 10
LR = 3e-4
BATCH_SIZE = 32
DECAY = 0.0001

best_accuracy = 0.0 

# DataLoader for train and test datasets
train_loader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=BATCH_SIZE)

# model 
loop = tqdm(train_loader)
model = CNN(1, 10)

# define loss function 
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = LR, weight_decay=DECAY)


# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)

# function to train a model
def train():
    for epoch in range(EPOCHS):
        running_loss = 0
        for i, (images, labels) in enumerate(loop):     
            
            # don't let the previous gradient disturb the present 
            # make the previous storage of gradient zero
            optimizer.zero_grad()

            outputs = model(images)
            
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy for this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
            
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

if __name__ == '__main__':
    train()

