
""" 
Author : Rohin Andy Ramesh
University of Bordeaux
course: IPCV 
4TTV911U Deep Learning in Computer Vision
Lab1 

"""

#just Run this file



import os
import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


classname = {
    0: 'Colonial',
    1: 'Modern',
    2: 'Prehispanic' 
}

def split_data(train_dataset_path,test_dataset_path):
    trainfilenames = os.listdir(train_dataset_path)
    testfiles = os.listdir(test_dataset_path)

    train_class_names = {}
    test_class_names = {}

    for txt in trainfilenames:
        spi = txt.split("_")
        image_file = os.path.join(train_dataset_path,txt)
        train_class_names[image_file] = spi[0]

    for txt in testfiles:
        spi = txt.split("_")
        image_file = os.path.join(test_dataset_path,txt)
        test_class_names[image_file] = spi[0]
    
    for i in train_class_names:
        if train_class_names[i] == 'Colonial':
            train_class_names[i] = 0
        elif  train_class_names[i] == 'Modern':
            train_class_names[i] = 1
        elif  train_class_names[i] == 'Prehispanic':
            train_class_names[i] = 2


    for i in test_class_names:
        if test_class_names[i] == 'Colonial':
            test_class_names[i] = 0
        elif  test_class_names[i] == 'Modern':
            test_class_names[i] = 1
        elif test_class_names[i] == 'Prehispanic':
            test_class_names[i] = 2


    return train_class_names , test_class_names

# Custom dataset class to handle your folder structure and filenames
class CustomDataset(Dataset):
    def __init__(self, file_dict, transform=None):
        self.file_dict = file_dict  # {filepath: class_name}
        self.filepaths = list(file_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.file_dict[img_path]

        # Load the image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(train_class_names,test_class_names):

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    # Initialize datasets
    train_dataset = CustomDataset(train_class_names, transform=data_transforms['train'])
    test_dataset = CustomDataset(test_class_names, transform=data_transforms['val'])

    # # Load data
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    dataset_sizes = {'train': 236,
                    'test':48}
    dataloaders = {'train': DataLoader(train_dataset, batch_size=4, shuffle=True),
                'val' : DataLoader(test_dataset, batch_size=4, shuffle=False)}


    return dataloaders 


def train_model(dataloaders, n_epochs):
    net = models.resnet18(pretrained=True)
    net = net.cuda() if device else net

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    def accuracy(out, labels):
        _,pred = torch.max(out, dim=1)
        return torch.sum(pred==labels).item()

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 128)
    net.fc = net.fc.cuda()

    n_epochs = n_epochs
    print_every = 10
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(dataloaders['train'])
    for epoch in range(1, n_epochs+1):
        running_loss = 0.0
        correct = 0
        total=0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(dataloaders['train']):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()

            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
            if (batch_idx) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
        batch_loss = 0
        total_t=0
        correct_t=0
        with torch.no_grad():
            net.eval()
            for data_t, target_t in (dataloaders['val']):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t/total_t)
            val_loss.append(batch_loss/len(dataloaders['val']))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')


            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), 'resnet.pt')
                print('Improvement-Detected, save-model')
        net.train()
    
    return val_loss ,val_acc ,train_loss ,train_acc ,net,criterion


def plotcurves(train_acc, val_acc, train_loss, val_loss):
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))

    # Plot accuracy
    axs[0].plot(train_acc, label='Train Accuracy')
    axs[0].plot(val_acc, label='Validation Accuracy')
    axs[0].set_title("Train-Validation Accuracy")
    axs[0].set_xlabel('Num Epochs', fontsize=12)
    axs[0].set_ylabel('Accuracy', fontsize=12)
    axs[0].legend(loc='best')

    # Plot loss
    axs[1].plot(train_loss, label='Train Loss')
    axs[1].plot(val_loss, label='Validation Loss')
    axs[1].set_title("Train-Validation Loss")
    axs[1].set_xlabel('Num Epochs', fontsize=12)
    axs[1].set_ylabel('Loss', fontsize=12)
    axs[1].legend(loc='best')

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('train_validation_curves.png')
    
    # Show the plot
    plt.show()

def plot_cfmat(dataloaders, net):
    # Collect predictions and true labels
    all_preds = []
    all_targets = []

    # Validation phase to collect predictions
    with torch.no_grad():
        net.eval()
        for data_t, target_t in dataloaders['val']:
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = net(data_t)
            _, pred_t = torch.max(outputs_t, dim=1)

            # Collect predictions and targets
            all_preds.extend(pred_t.cpu().numpy())
            all_targets.extend(target_t.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classname.values(), yticklabels=classname.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the figure
    plt.savefig('confusion_matrix.png')
    
    # Show the plot
    plt.show()



def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize the image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')  # Hide axis ticks



def plotmisclassified(dataloaders, net, criterion):
    incorrect_samples = []

    with torch.no_grad():
        net.eval()
        for data_t, target_t in dataloaders['val']:
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = net(data_t)
            loss_t = criterion(outputs_t, target_t)

            _, pred_t = torch.max(outputs_t, dim=1)

            # Find misclassified samples
            misclassified = (pred_t != target_t)

            # Store the misclassified samples, their actual and predicted labels, and probabilities
            if misclassified.any():
                for i in range(len(misclassified)):
                    if misclassified[i]:
                        predicted_prob = torch.softmax(outputs_t[i], dim=0)[pred_t[i]].item()
                        incorrect_samples.append({
                            'image': data_t[i].cpu(),
                            'actual': target_t[i].cpu().item(),
                            'predicted': pred_t[i].cpu().item(),
                            'probability': predicted_prob
                        })
    
    # Sort by probability (ascending, worst predictions first)
    incorrect_samples = sorted(incorrect_samples, key=lambda x: x['probability'])

    # Select the 10 worst classified images
    worst_samples = incorrect_samples[:10]

    # Plotting the 10 worst classified images
    fig = plt.figure(figsize=(20, 8))  # Increased size for better visibility

    for i, sample in enumerate(worst_samples):
        img = sample['image']
        actual_label = sample['actual']
        predicted_label = sample['predicted']
        probability = sample['probability']

        # Validate class labels
        actual_class = classname.get(actual_label, 'Unknown')
        predicted_class = classname.get(predicted_label, 'Unknown')

        # Add subplot for each image
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        imshow(img, f'Actual: {actual_class}\nPred: {predicted_class}\nProb: {probability:.2f}')

    plt.tight_layout()
    
    # Save the figure
    plt.savefig('misclassified_images.png')
    
    # Show the plot
    plt.show()


def main():
    train_dataset_path = "/net/ens/DeepLearning/DLCV2024/MexCulture142/images_train/"
    test_dataset_path  = "/net/ens/DeepLearning/DLCV2024/MexCulture142/images_val/"


    traindict , testdict = split_data(train_dataset_path,test_dataset_path)
    dataloaders  = get_dataloaders(traindict,testdict)
    val_loss ,val_acc ,train_loss ,train_acc ,net,criterion = train_model(dataloaders, 50)
    plotcurves(train_acc, val_acc, train_loss, val_loss)
    plot_cfmat(dataloaders, net)
    plotmisclassified(dataloaders,net,criterion)


if __name__ == '__main__':
    main()    


