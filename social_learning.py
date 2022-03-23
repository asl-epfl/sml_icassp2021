import numpy as np
import torch
import nn
from parameters import *
from generate_data import *
import matplotlib.pyplot as plt

#%%

def generate_sc_graph(num_agents):
    '''
    Generate strongly connected graph.

    Parameters
    ----------
    num_agents: int
        number of agents

    Returns
    -------
    G: ndarray
        graph adjacency matrix
    '''
    G = np.random.choice([0.0, 1.0], size=(num_agents, num_agents), p=[0.3, 0.7])
    G = G + np.eye(num_agents)
    G = (G > 0) * 1.0
    return G


def create_uniform_combination_matrix(G):
    '''
    Generate combination matrix using the uniform rule.

    Parameters
    ----------
    G: ndarray
        adjacency matrix

    Returns
    -------
    A: ndarray
        combination matrix
    '''
    A = G / np.sum(G, 0)
    return A

def create_graph_of_nn(num_agents, hidden_size, num_classes, mnist_input):
    '''
    Attribute the training models for each gent.

    Parameters
    ----------
    num_agents: int
        number of agents
    hidden_size: int
        size of the hidden layer for the NN
    num_classes: int
        output size for the NN
    mnist_input: int
        input size for the NN

    Returns
    -------
    N: list(nn.Module)
        list of all modules
    A: ndarray
        combination matrix
    '''
    G = generate_sc_graph(num_agents)
    A = create_uniform_combination_matrix(G)
    N = []
    for i in range(num_agents):
        N.append(nn.Net(mnist_input[i], hidden_size, num_classes))
        N[i].load_state_dict(torch.load('models/agent_{}.pkl'.format(i)))
        N[i].eval()
    return N, A


def asl(mu_0, d, test_loader, num_agents, N, A):
    '''
    Run prediction phase using the ASL algorithm.

    Parameters
    ----------
    mu_0: ndarray
        initial beliefs
    d: float
        step-size parameter
    test_loader: Dataloader
        test Dataloader
    num_agents: int
        number of agents
    N: list(nn.Module)
        list of NN models
    A: ndarray
        combination matrix

    Returns
    -------
    MU: list(ndarray)
        Belief (log-ratio) evolution over time
    '''
    mu = np.log(mu_0[:, 1] / mu_0[:, 0])
    MU = [mu]
    for i in range(len(test_loader[0])):
        L=[]
        for j in range(num_agents):
            feat = (test_loader[j].dataset.data[i]).float()
            feat = feat.view(1, -1)
            outputs = N[j](feat)
            L.append(outputs.detach().numpy())
        L = np.array(L)[:, 0]
        mu = (1-d) * A.T @ mu + d * A.T @ np.log(L[:,1] / L[:,0])
        MU.append(mu)
    return MU

