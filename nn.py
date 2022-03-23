from generate_data import *
import torch
import torch.nn as nn
from torch.autograd import Variable

#%%
class Net(nn.Module):
    '''
    Creates a sequence of linear layers and Tanh activations.

    Attributes
    ----------
    num_feats :  int
        int value indicating input size
    hidden_size : int
        int value indicating hidden layer size
    num_classes : int
        int value indicating output size

    Methods
    -------
    forward()
    '''
    def __init__(self, num_feats, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_feats, hidden_size, bias= True)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias= True)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


#%%
def train(num_epochs, train_load, train_ds, net, optimizer, criterion):
    '''
    Trains a model using one optimization criterion for a certain number of epochs.
    Returns training performance indicators over epochs.

    Parameters
    ----------
    num_epochs: int
        Number of training epochs
    train_load: DataLoader
        Training loader
    train_ds: Dataset
        Training dataset
    net: torch.nn.Module
        model to be trained
    optimizer: PyTorch optimizer
        PyTorch optimizer
    criterion: PyTorch criterion
        PyTorch criterion

    Returns
    -------
    loss_epochs: list(FloatTensor)
        training loss over epochs
    '''
    net = net.float()
    loss_epochs = []
    for epoch in range(num_epochs):
        for i, (feats, labels) in enumerate(train_load):
            feats = feats.view(feats.shape[0], -1)
            feats = Variable(feats.float())
            labels = Variable(labels)
            optimizer.zero_grad()
            outputs = net(feats)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_ds)//batch_size, loss.data))
            if i == 0:
                loss_epochs.append(loss)
    return loss_epochs
#%%
def test(test_load, net, criterion):
    '''
    Evaluate the model on the test dataset.

    Parameters
    ----------
    test_load: DataLoader
        Test loader
    net: torch.nn.Module
        model to be evaluated
    criterion: PyTorch criterion
        PyTorch criterion

    Returns
    -------
    acc: FloatTensor
        test accuracy
    _ : FloatTensor
        test loss
    '''
    net = net.float()
    correct = 0
    total = 0
    loss_test =[]
    for feats, labels in test_load:
        feats = feats.view(1, -1)
        outputs = net(feats.float())
        loss_test.append(criterion(outputs, labels.long()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = torch.true_divide(100 * correct, total)
    return accuracy, sum(loss_test)/len(loss_test)

#%%

def train_test_agent(id):
    '''
    Trains and test the model for one agent.
    Returns training and testing performance indicators over epochs.
    Returns the model and test loader to be used in the prediction phase.

    Parameters
    ----------
    id: int
        number of the agent to be trained

    Returns
    -------
    loss_epochs: list(FloatTensor)
        training loss over epochs
    acc: FloatTensor
        test accuracy
    ltest : FloatTensor
        test loss
    net: nn.Module
        trained model
    test_load: DataLoader
        Test loader
    '''
    train_load, test_load, train_ds, test_ds = generate_bin_MNIST(id, batch_size, num_agents)
    net = Net(mnist_input[id], hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_epochs = train(num_epochs, train_load, train_ds, net, optimizer, criterion)
    acc_test, ltest = test(test_load, net, criterion)
    return loss_epochs, acc_test, ltest, net, test_load
