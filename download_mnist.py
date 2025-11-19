from torchvision import datasets, transforms;
from torch.utils.data import random_split;
from torch.utils.data import DataLoader;

def get_MNIST_loaders(train_batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_size = 50000
    validation_size = 10000

    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, validation_loader, test_loader