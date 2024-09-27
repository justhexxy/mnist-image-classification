from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# Download the datasets if not downloaded
train_dataset = datasets.MNIST(root="./mnist_dataset", download=True, transform=transforms.ToTensor(), train=True)
test_dataset = datasets.MNIST(root="./mnist_dataset", download=True, transform=transforms.ToTensor(), train=False)
train_dataset, val_dataset = random_split(train_dataset, lengths=[50_000,10_000])

# Define DataLoaders

train_loader = DataLoader(
        dataset= train_dataset,
        batch_size = 512,
        shuffle = True
)

test_loader = DataLoader(
        dataset= test_dataset,
        batch_size = 512,
        shuffle = True
)

val_loader = DataLoader(
        dataset= val_dataset,
        batch_size = 512,
        shuffle = True
)

# Define Multi Layer Neural Network
class MNIST_Classifier(torch.nn.Module):
    """Simple Neural Network Model used to classify MNIST Dataset"""
    def __init__(self, num_features, num_classes):
        """
        The architecture is defined in the following way:
        """
        super().__init__()
        self.all_layers = torch.nn.Sequential(
                # First Layer
                torch.nn.Linear(num_features, 50),
                torch.nn.ReLU(),
                # Hidden layers
                torch.nn.Linear(50, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 25),
                torch.nn.ReLU(),
                # Output Layer
                torch.nn.Linear(25, num_classes)
        )

    def forward(self, x):
        """
        Forward Propagation method that returns the calculations of the
        architecture defined in __init__ method
        """
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits

def compute_accuracy(model, dataloader):
    model = model.eval() # Turn model into evaluate mode
    correct = 0
    total_examples = 0
    for idx, (features, labels) in enumerate(dataloader):
        with torch.inference_mode():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        result = (predictions == labels)
        correct += torch.sum(result)
        total_examples += len(result) # ???
    return correct / total_examples

# Training Loop
torch.manual_seed(1)
model = MNIST_Classifier(784, num_classes=10)
acc = compute_accuracy(model, val_loader)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 20
loss_list = []
train_acc_list, val_acc_list = [],[]

for epoch in tqdm(range(num_epochs)):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = torch.nn.functional.cross_entropy(logits,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_accuracy = compute_accuracy(model,val_loader )
    train_accuracy = compute_accuracy(model, train_loader)
    train_acc_list.append(train_accuracy)
    val_acc_list.append(val_accuracy)
    print(f"in epoch {epoch} loss is: {loss.detach()}, train_acc={train_accuracy}, val_acc={val_accuracy}")

# Evaluate
val_accuracy = compute_accuracy(model,val_loader )
train_accuracy = compute_accuracy(model, train_loader)
test_accuracy =  compute_accuracy(model, test_loader)

print(f"Train: {train_accuracy}")
print(f"Validation: {val_accuracy}")
print(f"Test: {test_accuracy}")
