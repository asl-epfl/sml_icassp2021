import numpy as np
import torch
from parameters import *
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#%%
def balance_two_datasets(train_0, train_1):
    '''
    Balance two datasets.

    Parameters
    ----------
    train_0: FloatTensor
        dataset 1
    train_1: FloatTensor
        dataset 2

    Returns
    -------
    train_0: FloatTensor
        balanced dataset 1
    train_1: FloatTensor
        balanced dataset 2

    '''
    d_train = len(train_1) - len(train_0)
    if d_train > 0:
        train_1 = train_1[:-d_train]
    elif d_train < 0:
        train_0 = train_0[:d_train]
    return train_0, train_1

def generate_bin_MNIST(id, batch_size, num_agents):
    '''
    Generate balanced MNIST training/ testing Datasets and Dataloaders for one agent.

    Parameters
    ----------
    id: int
        index of agent
    batch_size: int
        size of batches

    Returns
    -------
    trainloader: DataLoader
        training DataLoader
    testloader: DataLoader
        test DataLoader
    trainset: Dataset
        training Dataset
    testset: Dataset
        test Dataset

    '''
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    testset = datasets.MNIST('mnist_train', download=True, train=True, transform=transform)
    trainset = datasets.MNIST('mnist_test', download=True, train=False, transform=transform)

    train_0 = trainset.data[np.argwhere(trainset.targets==0).numpy()]
    train_1 = trainset.data[np.argwhere(trainset.targets==1).numpy()]

    test_0 = testset.data[np.argwhere(testset.targets==0).numpy()]
    test_1 = testset.data[np.argwhere(testset.targets==1).numpy()]

    train_0, train_1 = balance_two_datasets(train_0, train_1)
    n_train = len(train_0)
    N_train_ag = n_train//num_agents

    test_0, test_1 = balance_two_datasets(test_0, test_1)
    n_test = len(test_0)
    N_test_ag = N_TEST//2

    if id==0:
        trainset.data = torch.cat((train_1[id*N_train_ag:(id+1)*N_train_ag],
                                   train_1[(id+1)*N_train_ag:(id+2)*N_train_ag]), 0)
        testset.data = torch.cat((test_0[id*N_test_ag:(id+1)*N_test_ag],
                                  test_1[id*N_test_ag:(id+1)*N_test_ag]), 0)

        trainset.targets = torch.cat((torch.ones(N_train_ag),
                                      torch.zeros(N_train_ag)))
        testset.targets = torch.cat((torch.zeros(N_test_ag),
                                     torch.ones(N_test_ag)))
    else:
        trainset.data = torch.cat((train_0[id*N_train_ag:(id+1)*N_train_ag],
                                   train_1[id*N_train_ag:(id+1)*N_train_ag]), 0)
        testset.data = torch.cat((test_0[id*N_test_ag:(id+1)*N_test_ag],
                                  test_1[id*N_test_ag:(id+1)*N_test_ag]), 0)

        trainset.targets = torch.cat((torch.zeros(N_train_ag),
                                      torch.ones(N_train_ag)))
        testset.targets = torch.cat((torch.zeros(N_test_ag),
                                     torch.ones(N_test_ag)))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    return trainloader, testloader, trainset, testset
