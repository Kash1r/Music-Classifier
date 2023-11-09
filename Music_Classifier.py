# Import the required libraries.
# torch is the main PyTorch library, torchvision is a library that
# contains convenient utilities for working with image data in PyTorch.
# torchaudio is a library that contains tools for working with audio data.
# numpy is a numerical computing library, and pandas is a data manipulation library.
# os and pathlib are used for file and directory operations.
# matplotlib is a plotting library, and PIL is an image manipulation library.
import torch
import torchvision
from torchvision import models
import random
import librosa.display
import os
import pathlib
import matplotlib.pyplot as plt

# Import PyTorch's neural network, functional, transforms, and data loader modules.
# The neural network module contains the building blocks for our neural networks.
# The functional module contains function-style versions of various operations
# The transforms module allows us to perform transformations on our dataset.
# The data loader module allows us to load data in a particular way, which is especially useful when dealing with batches of images.
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

# tqdm is a library that allows us to create progress bars.
from tqdm.autonotebook import tqdm

# torchvision.transforms allows us to perform transformations on our dataset.
import torchvision.transforms as T

# The directory where the music files are stored.
data_path = '/content/genres/'

# Convert music files to spectrogram images using librosa and save them in a folder.
# Here we're setting the color map for our plots to 'inferno' which is a type of color map.
cmap = plt.get_cmap('inferno')

# This is setting the size of the plot to 8x8 inches.
plt.figure(figsize=(8,8))

# These are the genres of the music files we're working with.
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# This loop goes through each genre, creates a directory for each one,
# goes through each file in the genre's directory, loads the audio file,
# creates a spectrogram for it, and saves the spectrogram as an image.
for g in genres:
    # Create the directory for the genre if it doesn't exist.
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)

    # Go through each file in the genre's directory.
    for filename in os.listdir(f'{data_path}/{g}'):

        # The path to the song.
        songname = f'{data_path}/{g}/{filename}'

        # Load the audio file. The file is loaded with mono=True so it's converted to mono (1 channel),
        # meaning the left and right channels are combined. The duration=5 parameter means only the first
        # 5 seconds of the file are loaded.
        y, sr = librosa.load(songname, mono=True, duration=5)

        # Create the spectrogram. The parameters are default except for the colormap and the scaling.
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')

        # Don't display any axes.
        plt.axis('off')

        # Save the spectrogram as a .png image.
        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')

        # Clear the current figure's content.
        plt.clf()

# The path to the folder where the spectrogram images are saved.
img_path = 'img_data'

# The batch size for the data loaders.
batch_size = 8

# The size to resize the images to.
image_size = 224

# The transformations to be applied to the images for the training set.
# These transformations will randomize the data a bit which can help improve training.
train_trms = T.Compose([
                        T.Resize(image_size),  # Resize the image to a specific size.
                        T.RandomRotation(20),  # Randomly rotate the image by a certain degree.
                        T.RandomHorizontalFlip(),  # Randomly flip the image horizontally.
                        T.ToTensor()  # Convert the image to a PyTorch tensor.
                        ])

# The transformations to be applied to the images for the validation set.
val_trms = T.Compose([
                        T.Resize(image_size),  # Resize the image to a specific size.
                        T.ToTensor()  # Convert the image to a PyTorch tensor.
                        ])

# Load the training data from the folder and apply the transformations.
train_data = torchvision.datasets.ImageFolder(root = img_path, transform = train_trms)

# Load the validation data from the folder and apply the transformations.
val_data = torchvision.datasets.ImageFolder(root = img_path, transform = val_trms)

# Function to encode the class labels into a dictionary mapping from index to class name.
def Encode(data):
    classes = data.classes
    encoder = {}
    for i in range(len(classes)):
        encoder[i] = classes[i]
    return encoder

# Function to decode the class labels into a dictionary mapping from class name to index.
def Decoder(data):
    classes = data.classes
    decoder = {}
    for i in range(len(classes)):
        decoder[classes[i]] = i
    return decoder

# Function to display a number of images along with their labels.
def class_plot(data, n_figures = 12):
    n_row = int(n_figures / 4)
    fig, axes = plt.subplots(figsize=(14, 10), nrows=n_row, ncols=4)
    for ax in axes.flatten():
        a = random.randint(0, len(data))
        (image, label) = data[a]
        label = int(label)
        encoder = Encode(data)
        l = encoder[label]
        image = image.numpy().transpose(1, 2, 0)
        im = ax.imshow(image)
        ax.set_title(l)
        ax.axis('off')
    plt.show()

# Calculate the size of the validation set as 10% of the training data.
val_size = int(len(train_data)*0.1)

# The rest of the data will be the training data.
train_size = len(train_data) - val_size

# Split the training data into training and validation sets.
train_ds, val_ds = random_split(train_data, [train_size,val_size])

# Create data loaders for the training and validation sets.
# This will allow us to load data in batches.
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

# Function to calculate the accuracy of the predictions.
def accuracy(outputs, labels):
  _,preds = torch.max(outputs,dim=1)
  return torch.tensor(torch.sum(preds == labels).item()/len(preds))

# Define the base class for the model.
class MultilabelImageClassificationBase(nn.Module):
    # This method calculates the loss for the training step.
    def training_step(self, batch):
        images, targets = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, targets)  # Calculate loss
        return loss

    # This method calculates the loss and score for the validation step.
    def validation_step(self, batch):
        images, targets = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, targets)  # Calculate loss
        score = accuracy(out, targets)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_score': score.detach()}

    # This method averages the losses and scores over the epoch.
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    # This method prints the loss and score at the end of each epoch.
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['val_loss'], result['val_score']))

# Define the custom model.
class Net1(MultilabelImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # This block is the first convolutional block.
            # It contains a convolutional layer, a ReLU activation function, and a max pooling layer.
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # This is the second convolutional block.
            # It's the same as the first, but it has 64 output channels.
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # This is the third convolutional block.
            # It's the same as the second.
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # This is the fourth convolutional block.
            # It's the same as the third, but it has 128 output channels.
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # This is the fifth convolutional block.
            # It's the same as the fourth.
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # This is the sixth convolutional block.
            # It's the same as the fifth, but it has 256 output channels.
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            # This layer flattens the output of the previous layer into a one dimensional tensor.
            nn.Flatten(),

            # These are linear (fully connected) layers.
            # They map the features learned by the convolutional layers to the final output classes.
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    # This method is used to pass an input through the model.
    def forward(self, xb):
        return self.network(xb)

# Define the model with a pretrained network.
class Net(MultilabelImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model.
        self.network = models.resnet34(pretrained=True)
        # Replace last layer.
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 10)

    # This method is used to pass an input through the model.
    def forward(self, xb):
        return self.network(xb)

    # This method freezes the parameters in the model.
    def freeze(self):
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True

    # This method unfreezes the parameters in the model.
    def unfreeze(self):
        for param in self.network.parameters():
            param.require_grad = True

# Function to evaluate the model.
def evaluate(model,val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# Function to get the learning rate from the optimizer.
@torch.no_grad()
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Function to fit the model.
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    # Set up cutom optimizer with weight decay.
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler.
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase.
        model.train()
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            loss.backward()
            # Gradient clipping.
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            # Record & update learning rate.
            lrs.append(get_lr(optimizer))
            sched.step()
        # Validation phase.
        result = evaluate(model, val_loader)
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# Function to pick GPU if available, else CPU.
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# Function to move tensor(s) to chosen device.
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Class to wrap a dataloader to move data to a device.
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

# Get the device.
device = get_default_device()

# Move the dataloaders to the device.
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

# Move the model to the device.
model = to_device(Net(), device)

# Empty the CUDA cache.
torch.cuda.empty_cache()

# Evaluate the model initially.
history = [evaluate(model, val_dl)]

# Freeze the model.
model.freeze()

# Define the training parameters.
epochs = 5
max_lr = 0.001
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

# Fit the model.
history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                         grad_clip=grad_clip,
                         weight_decay=weight_decay,
                         opt_func=opt_func)

# Save the model parameters.
torch.save(model.state_dict(), '/content/model.pth')