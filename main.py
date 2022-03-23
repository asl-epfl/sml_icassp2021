from nn import *
from parameters import *
import matplotlib.pyplot as plt
from social_learning import *
import networkx as nx
import os
#%%
plt.style.use('seaborn-colorblind')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble":r"\usepackage{bm}"
})

#%%
FIG_PATH = 'figs/'
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)

DATA_PATH = 'data/'
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

MODELS_PATH = 'models/'
if not os.path.isdir(MODELS_PATH):
    os.makedirs(MODELS_PATH)

#%%
def train():
    '''
    Train models. Save models in a dedicated folder.
    Save training performance for plotting.
    '''
    np.random.seed(SEED)
    loss_ag, acc_test_ag, ltest_ag, test_loader = [], [], [], []
    for l in range(num_agents):
        loss_epochs, acc_test, ltest, net, test_load = train_test_agent(l)
        test_loader.append(test_load)
        loss_ag.append(loss_epochs)
        acc_test_ag.append(acc_test)
        ltest_ag.append(ltest)
        torch.save(net.state_dict(), MODELS_PATH + 'agent_{}.pkl'.format(l))
    torch.save(test_loader, DATA_PATH + 'test_agents.pkl')
    torch.save((loss_ag, acc_test_ag, ltest_ag), DATA_PATH + 'train_stats.pkl')

def test_sl():
    '''
    Simulate prediction phase using trained models.
    Output beliefs are saved for plotting.
    '''
    np.random.seed(SEED)
    test_loader = torch.load(DATA_PATH + 'test_agents.pkl')

    mu_0 = np.random.rand(num_agents, num_classes)
    mu_0 = mu_0 / np.sum(mu_0, axis=1)[:, None]
    d = 0.1
    Net, A = create_graph_of_nn(num_agents, hidden_size, num_classes, mnist_input)
    MU = asl(mu_0, d, test_loader, num_agents, Net, A)
    torch.save((MU, A), DATA_PATH + 'sl.pkl')

def plot_sl():
    '''
    Plot network topology, training performance and belief evolution during prediction.
    Save figures in dedicated folder.
    '''
    MU, A = torch.load(DATA_PATH + 'sl.pkl')
    Gr = nx.from_numpy_matrix(A)
    pos = nx.spring_layout(Gr, seed=SEED)
    colors = plt.cm.gray(np.linspace(0.3, .8, num_agents))

    # Figure 1: Network Topology
    f, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    plt.axis('off')
    plt.xlim([-1, 1.15])
    plt.ylim([-1.2, .85])
    nx.draw_networkx_nodes(Gr,
                           pos=pos,
                           node_color='C0',
                           nodelist=[0],
                           node_size=300,
                           edgecolors='k',
                           linewidths=0.5)
    nx.draw_networkx_nodes(Gr,
                           pos=pos,
                           node_color=colors[8],
                           nodelist=range(1, num_agents),
                           node_size=300,
                           edgecolors='k',
                           linewidths=.5)
    nx.draw_networkx_labels(Gr,
                            pos=pos,
                            labels={i: i + 1 for i in range(num_agents)},
                            font_size=14,
                            font_color='black',
                            alpha=1)
    nx.draw_networkx_edges(Gr,
                           pos=pos,
                           node_size=300,
                           alpha=1,
                           arrowsize=6,
                           width=1);

    plt.savefig(FIG_PATH + 'net.pdf', bbox_inches='tight')

    # Compute Perron eigenvector
    e,v = np.linalg.eig(A)
    pv = np.real(v[:, np.where(np.isclose(e, 1))[0]])[:, 0]
    pv = pv / np.sum(pv)

    # Figure 2: Training performance
    loss_ag, acc_test_ag, ltest_ag = torch.load(DATA_PATH + 'train_stats.pkl')
    plt.figure(figsize=(4,2.5))
    plt.plot(np.arange(1, num_epochs + 1),
             loss_ag[0],
             '--o',
             color='C0',
             label='Agent 1',
             markersize=4)
    for i in range(1,num_agents):
        plt.plot(np.arange(1, num_epochs + 1),
                 loss_ag[i],
                 '--o',
                 color=colors[i],
                 markersize=4)

    plt.plot(np.arange(1, num_epochs + 1),
             pv@np.array(loss_ag),
             '--o',
             color='C2',
             label='Network Avg.',
             markersize=4)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Empirical risk', fontsize=14)
    plt.xlim([1,num_epochs])
    plt.ylim([.3,.75])
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.legend( fontsize=14,loc = 'center right',handlelength=1)
    plt.savefig(FIG_PATH + 'risk.pdf', bbox_inches='tight')

    # Print prediction accuracy and prediction cost:
    for i in range(num_agents):
        print('Agent %d Test accuracy %d %% and risk %1.3f'%(i, acc_test_ag[i], ltest_ag[i]))

    # Figure 3: Prediction performance
    plt.figure(figsize=(4, 2.5))
    plt.plot(np.arange(N_TEST+1), [MU[i][0] for i in range(len(MU))],
             linewidth=1.5)
    plt.plot(np.arange(N_TEST+1),np.zeros(N_TEST+1),
             linewidth=1.5,
             color='k',
             linestyle='dashed')
    plt.text(780, 1, 'Digit 1', fontsize=13)
    plt.text(780, -3, 'Digit 0', fontsize=13)
    plt.ylabel(r'$\bm{\lambda}_{{%d},i}$' %(0+1), fontsize=16, labelpad=-11)
    plt.xlabel(r'$i$', fontsize=16,labelpad=2)
    plt.xticks(np.arange(0, len(MU), 500))
    plt.xlim([0, len(MU)])
    plt.ylim([-12, 12])
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.savefig(FIG_PATH + 'sl_agent1.pdf', bbox_inches='tight')

#%%
if __name__ == '__main__':
    train()

    test_sl()

    plot_sl()


