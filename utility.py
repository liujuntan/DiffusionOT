import torch
torch.cuda.empty_cache()
import torch.nn as nn
import numpy as np
from TorchDiffEqPack import odesolve
import sys
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import random
from torchdiffeq import odeint
from functools import partial
import getpass
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns
import warnings
from sklearn.mixture import GaussianMixture
import pandas as pd
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap, Normalize
import joblib
import scanpy as sc
  
warnings.filterwarnings("ignore")

class Args:
    pass

def create_args():
    args = Args()
    args.dataset = input("Name of the data set. Options: EMT; Mouse; Zebrafish; Spatial; MISA (default: MISA): ") or 'MISA'
    timepoints = input("Time points of data (default: 0,1,2,6): ")
    args.timepoints = [float(tp.strip()) for tp in timepoints.split(",")] if timepoints else [0,1,2,6]
    args.niters = int(input("Number of training iterations (default: 5000): ") or 5000)
    args.lr = float(input("Learning rate (default: 3e-3): ") or 3e-3)
    args.num_samples = int(input("Number of sampling points per epoch (default: 100): ") or 100)
    args.hidden_dim = int(input("Dimension of the hidden layer (default: 16): ") or 16)
    args.n_hiddens = int(input("Number of hidden layers (default: 4): ") or 4)
    args.activation = input("Activation function (default: Tanh): ") or 'Tanh'
    args.gpu = int(input("GPU device index (default: 0): ") or 0)
    args.input_dir = input("Input Files Directory (default: Input/): ") or 'Input/'
    args.save_dir = input("Output Files Directory (default: Output/): ") or 'Output/'
    args.seed = int(input("Random seed (default: 1): ") or 1)
    args.d = float(input("Diffusion coefficient (default: 0.001): ") or 0.001)
    return args


class RUOT(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation, d):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.hyper_net1 = HyperNetwork1(in_out_dim, hidden_dim, n_hiddens,activation) #v= dx/dt
        self.hyper_net2 = HyperNetwork2(in_out_dim, hidden_dim, activation) #g
        self.hyper_net3 = HyperNetwork1(in_out_dim, hidden_dim, n_hiddens,activation) #score
        self.d = nn.Parameter(torch.tensor(d), requires_grad=True)  # Set d as a trainable parameter with gradient descent

    def forward(self, t, states):
        z = states[0]
        batchsize = z.shape[0]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            dz_dt = self.hyper_net1(t, z)
            g = self.hyper_net2(t, z)
            dlog_p_dz = self.hyper_net3(t, z)##
            dlogp_z_dt = g - trace_df_dz(dz_dt, z).view(batchsize, 1)##
          
        return (dz_dt, g, dlogp_z_dt, dlog_p_dz)##?


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()

class HyperNetwork1(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation='Tanh'):
        super().__init__()
        Layers = [in_out_dim+1]
        for i in range(n_hiddens):
            Layers.append(hidden_dim)
        Layers.append(in_out_dim)
        
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
        )
        self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]
        t = torch.tensor(t).repeat(batchsize).reshape(batchsize, 1)
        t.requires_grad=True
        state  = torch.cat((t,x),dim=1)
        
        ii = 0
        for layer in self.net:
            if ii == 0:
                x = layer(state)
            else:
                x = layer(x)
            ii =ii+1
        x = self.out(x)
        return x
    
class HyperNetwork2(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        batchsize = x.shape[0]
        t = torch.tensor(t).repeat(batchsize).reshape(batchsize, 1)
        t.requires_grad=True
        state  = torch.cat((t,x),dim=1)
        return self.net(state)
      
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def MultimodalGaussian_density(x,time_all,time_pt,data_train,sigma,device):
    """density function for MultimodalGaussian
    """
    mu = data_train[time_all[time_pt]]
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim).type(torch.float32).to(device)
    p_unn = torch.zeros([x.shape[0]]).type(torch.float32).to(device)
    for i in range(num_gaussian):
        m = torch.distributions.multivariate_normal.MultivariateNormal(mu[i,:], sigma_matrix)
        p_unn = p_unn + torch.exp(m.log_prob(x)).type(torch.float32).to(device)
    p_n = p_unn/num_gaussian
    return p_n

def MultimodalGaussian_density_sample(x,time_all,time_pt,data_train,sigma,device,sample_size=500):
    """density function for MultimodalGaussian by sampling
    """
    mu = data_train[time_all[time_pt]]
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim).type(torch.float32).to(device)
    p_unn = torch.zeros([x.shape[0]]).type(torch.float32).to(device)
    
    # Randomly sample 'sample_size' indices from the total number of samples
    sample_indices = torch.randint(0, num_gaussian, (sample_size,)).to(device)
    sampled_mu = mu[sample_indices]
    
    for i in range(sample_size):  # Loop over the sampled indices
        m = torch.distributions.multivariate_normal.MultivariateNormal(sampled_mu[i, :], sigma_matrix)
        p_unn = p_unn + torch.exp(m.log_prob(x)).type(torch.float32).to(device)
    
    p_n = p_unn / sample_size
    
    return p_n
   
def Sampling(num_samples,time_all,time_pt,data_train,sigma,device):
    #perturb the  coordinate x with Gaussian noise N (0, sigma*I )
    mu = data_train[time_all[time_pt]]
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim)#单位矩阵
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), sigma_matrix)
    noise_add = m.rsample(torch.Size([num_samples])).type(torch.float32).to(device)
    # check if number of points is <num_samples
    if num_gaussian < num_samples:
        samples = mu[random.choices(range(0,num_gaussian), k=num_samples)] + noise_add
    else:
        samples = mu[random.sample(range(0,num_gaussian), num_samples)] + noise_add
    return samples

def calculate_importance_weights(samples, time_all,time_pt,data_train,sigma,device):
    # Calculate density of each sample (e.g., using kernel density estimation)
    densities = MultimodalGaussian_density_sample(samples, time_all,time_pt,data_train,sigma,device) #normalized density(samples)
    # Invert the densities to get importance weights
    max_density = max(densities)
    weights = [max_density / density for density in densities]
    
    return weights

def ImportanceSampling(num_samples, time_all, time_pt, data_train, sigma, device):
    #perturb the  coordinate x with Gaussian noise N (0, sigma*I )
    mu = data_train[time_all[time_pt]]
    num_gaussian = mu.shape[0] # mu is number_sample * dimension
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim)#单位矩阵
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), sigma_matrix)
    noise_add = m.rsample(torch.Size([num_samples])).type(torch.float32).to(device)
    # check if number of points is <num_samples
    # Calculate importance weights based on the density of the samples
    weights = calculate_importance_weights(mu, time_all,time_pt,data_train,1,device)
    if num_gaussian < num_samples:
        samples = mu[random.choices(range(0,num_gaussian), k=num_samples, weights=weights)] + noise_add
    else:
        samples = mu[random.choices(range(0,num_gaussian), k=num_samples, weights=weights)] + noise_add
    return samples
    
def add_noise(points,device,sigma=0.01):
    #noise = torch.randn_like(points) * variance ** 0.5
    dim = points.shape[1]
    sigma_matrix = sigma * torch.eye(dim)#单位矩阵
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), sigma_matrix)
    noise_add = m.rsample(torch.Size([len(points)])).type(torch.float32).to(device)
    return points + noise_add

def loaddata(args,device):
    data=np.load(os.path.join(args.input_dir,(args.dataset+'.npy')),allow_pickle=True)
    data_train=[]
    for i in range(data.shape[1]):
        data_train.append(torch.from_numpy(data[0,i]).type(torch.float32).to(device))
    return data_train

def load_data(dataset:str,path_to_data):
    if dataset=='EMT':
        adata = sc.read_h5ad(path_to_data+'emt.h5ad')
        X=adata.X

    elif dataset=='Mouse':
        adata = sc.read_h5ad(path_to_data+'mouse_pre.h5ad')
        adata.obs.rename(columns={'Time point': 'time','Cell type annotation':'cell type'}, inplace=True)
        adata.obs['time'] = adata.obs['time'].astype(str)
        X=adata.X

    else:
        raise NotImplementedError
    return adata,X

def ggrowth(t,y,func,device):
    y_0 = torch.zeros(y[0].shape).type(torch.float32).to(device)
    y_00 = torch.zeros(y[1].shape).type(torch.float32).to(device)
    gg=func.forward(t, y)[1] 
    return (y_0,y_00,gg,y_0)
       
def trans_loss(t,y,func,device,odeint_setp,integral_time,train_time,data_train,sigma_now):#,GMM_g
    outputs= func.forward(t, y)
    v = outputs[0]
    g = outputs[1]
    dlog_p_dz = outputs[3]
    start_time,end_time=0,1
    for i in range(len(integral_time)-1):
      if integral_time[i]<=t<integral_time[i+1]:
          start_time=i
          end_time=i+1
          break
      elif t==integral_time[-1]:
          start_time=len(integral_time)-2
          end_time=len(integral_time)-1
          break
      elif t>integral_time[-1]:
          print("The time t should less than {}".format(integral_time[-1]))
          break
    
    prob_density1 = MultimodalGaussian_density_sample(y[0], train_time, start_time, data_train,sigma_now,device)
    prob_density2 = MultimodalGaussian_density_sample(y[0], train_time, end_time, data_train,sigma_now,device)
    p_i=(integral_time[end_time]-t)/(integral_time[end_time]-integral_time[start_time])
    p=p_i*prob_density1 + (1-p_i)*prob_density2 
    y_0 = torch.zeros(v.shape[0],1).type(torch.float32).to(device)
    y_00 = torch.zeros(v.shape).type(torch.float32).to(device)
    g_growth = partial(ggrowth,func=func,device=device)
    batchsize = y[0].shape[0]
    p=p.view(batchsize, 1).type(torch.float32)
    if torch.is_nonzero(t):
        _,_,exp_g,_ = odeint(g_growth, (y_00,y_0,y_0,y_00), torch.tensor([0,t]).type(torch.float32).to(device),atol=1e-5,rtol=1e-5,method='midpoint',options = {'step_size': odeint_setp})
        f_int = ((torch.norm(v,dim=1)**2+func.d**2*torch.norm(dlog_p_dz,dim=1)**2).view(batchsize, 1)-(2*func.d*torch.mul(g,torch.log(p)+1))+(torch.norm(g,dim=1)**2).view(batchsize, 1))*torch.exp(exp_g[-1])#alpha=1?.unsqueeze(1)
        return (y_00,y_0,f_int,y_00)
    else:
        return (y_00,y_0,y_0,y_00)

def score_f(t,y,func,device):
    y_0 = torch.zeros(y[0].shape).type(torch.float32).to(device)
    y_00 = torch.zeros(y[1].shape).type(torch.float32).to(device)
    batchsize = y[0].shape[0]##?
    dlog_p_dz=func.forward(t, y)[1]            
    gg = func.d*((torch.norm(dlog_p_dz,dim=1)**2).view(batchsize, 1)+trace_df_dz(dlog_p_dz, y[0]).view(batchsize, 1))##?
    return (y_0,y_00,gg)
    
def score_loss(t,y,func,device,odeint_setp,t0):
    outputs= func.forward(t, y)
    v = outputs[0]
    dlog_p_dz = outputs[3]
    y_0 = torch.zeros(v.shape[0],1).type(torch.float32).to(device)
    y_00 = torch.zeros(v.shape).type(torch.float32).to(device)
    batchsize = y[0].shape[0]
    if t!=t0:
        f_int = ((trace_df_dz(dlog_p_dz, y[0]).view(batchsize, 1))+(0.5*torch.norm(dlog_p_dz,dim=1)**2).view(batchsize, 1))*1#torch.exp(exp_g[-1])#alpha=1?
        return (y_00,y_0,f_int,y_00)
    else:
        return (y_00,y_0,y_0,y_00)

def gcd_list(numbers):
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    gcd_value = numbers[0]
    for i in range(1, len(numbers)):
        gcd_value = _gcd(gcd_value, numbers[i])
       
    return gcd_value

def pre_train_score(mse,func,args,data_train,train_time,integral_time,sigma_now,device,itr):
    warnings.filterwarnings("ignore")
    loss=0
    odeint_setp = gcd_list([num * 100 for num in integral_time])/100
    # compute score cost efficiency
    for i in range(len(train_time)-1):  
        #print(i)
        score_cost = partial(score_loss,func=func,device=device,odeint_setp=odeint_setp,t0=torch.tensor(integral_time[i]).type(torch.float32).to(device))
        x0 = Sampling(args.num_samples,train_time,i,data_train,sigma_now,device) 
        logp_diff_t00 = torch.zeros(x0.shape[0], 1).type(torch.float32).to(device)
        g_t00 = torch.zeros_like(x0).type(torch.float32).to(device)
        _,_,loss2,_= odeint(score_cost,y0=(x0, logp_diff_t00,logp_diff_t00,g_t00),t = torch.tensor([integral_time[i], integral_time[-1]]).type(torch.float32).to(device),atol=1e-5,rtol=1e-5,method='midpoint',options = {'step_size': odeint_setp})
        loss = loss + loss2[-1].mean(0)#*(integral_time[i]-integral_time[0])/(integral_time[-1]-integral_time[0])#+np.sum(const[i:])
        
    if (itr >1):
        if ((itr % 100 == 0) and (itr<=args.niters-500) and (sigma_now>0.0001) ):
            sigma_now = sigma_now/2
            
    return loss
    
def diffusion_fit(func,args,data_train,train_time,integral_time,device,time_tt=0.01):
    warnings.filterwarnings("ignore")
    viz_samples = 100
    sigma_a = 0.001
    n=10
    t_list2=[]
    
    integral_time2 = np.arange(integral_time[0], integral_time[-1]+time_tt, time_tt)
    integral_time2 = np.round_(integral_time2, decimals = 2)
    plot_time = list(reversed(integral_time2))
    
    with torch.no_grad():
        for i in range(len(integral_time)):

            z_t0 =  data_train[i]
            t_list2.append(integral_time[i])
        
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': None})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-5})
        options.update({'print_neval': False})
        options.update({'neval_max': 1000000})
        options.update({'safety': None})
        options.update({'t0': integral_time[-1]})
        options.update({'t1': 0})
        options.update({'t_eval':plot_time})
        D_t = torch.zeros((viz_samples, z_t0.shape[1])).to(device)
        for _ in range(n):
          #z_t0 =  Sampling(viz_samples, train_time,len(integral_time)-1,data_train,sigma_a,device)#len(train_time)-1
          z_t0 =  ImportanceSampling(viz_samples, train_time,len(integral_time)-1,data_train,sigma_a,device)
          logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
          g0 = torch.zeros(z_t0.shape).type(torch.float32).to(device)
          z_t1,_, logp_diff_t1,_= odesolve(func,y0=(z_t0,logp_diff_t0,logp_diff_t0,g0),options=options)
          D_t=D_t+torch.sum((z_t1[1:]-z_t1[0:-1])**2,dim=0)/(2*time_tt*(len(plot_time)-1))
          
        D_t=D_t/n 
    return D_t

def plot_diffusion(func,args,data_train,train_time,integral_time,device,D,time_tt=0.01):
    D_t=diffusion_fit(func,args,data_train,train_time,integral_time,device,time_tt)
    D_t=D_t.cpu().detach().numpy()
    plt.figure(figsize=(8, 6))
    # plot boxplot
    sns.boxplot(data=D_t)
    plt.axhline(D, color='r', linestyle='--')
    plt.title('Boxplot of estimated diffusion coefficient by dimensions')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.show()
    

# calculate distance matrix
def euclidean_distance(x, data):
    return ((x.unsqueeze(1) - data.unsqueeze(0)) ** 2).sum(dim=2)  

def find_nearest_points(x, data):
    distances = euclidean_distance(x, data)
    _, nearest_indices = distances.min(dim=1)
    return data[nearest_indices]

def True_v(data):
    x = data[:,0]
    y = data[:,1]
    # Convert float values to PyTorch tensors
    x = torch.tensor(x)
    y = torch.tensor(y)
    s = torch.tensor(0.5)

    a=0.7
    b=0.8
    n=4
    k=1.0
    dx= torch.div(a*x.pow(n),(s.pow(n)+x.pow(n)))+torch.div(b*s.pow(n),(s.pow(n)+y.pow(n)))-k*x
    dy= torch.div(a*y.pow(n),(s.pow(n)+y.pow(n)))+torch.div(b*s.pow(n),(s.pow(n)+x.pow(n)))-k*y
    return torch.stack((dx,dy),dim=1)

def euclidean_distance2(point1, point2):
    return torch.sqrt(torch.sum((point1 - point2) ** 2))

def group_distance2(group1, group2):
    distances = []
    for point1 in group1:
        for point2 in group2:
            distances.append(euclidean_distance2(point1, point2))
    
    distances_tensor = torch.tensor(distances, dtype=torch.float32)
    
    return torch.mean(distances_tensor)  
def group_distance(group1, group2):
    group1 = group1.unsqueeze(1)  # shape: [N, 1, D]
    group2 = group2.unsqueeze(0)  # shape: [1, M, D]
    diffs = group1 - group2  # shape: [N, M, D]
    distances = torch.sqrt(torch.sum(diffs ** 2, dim=-1))  # shape: [N, M]
    mean_distance = torch.mean(distances)  # scalar
    return mean_distance

def pre_train_model(mse,func,args,data_train,train_time,integral_time,sigma_now,options,device,itr):
    warnings.filterwarnings("ignore")

    loss = 0
    L2_value3 = torch.zeros(1,len(data_train)-1).type(torch.float32).to(device)
    L2_value4 = torch.zeros(1,len(data_train)-1).type(torch.float32).to(device)
    for i in range(len(data_train)-1): 
        
        x = Sampling(args.num_samples, train_time,i+1,data_train,0.0001,device)#0.02方差
        x.requires_grad=True
        logp_diff_t1 = torch.zeros(x.shape[0], 1).type(torch.float32).to(device)
        g_t1 = torch.zeros_like(x).type(torch.float32).to(device)
        # loss between each two time points
        options.update({'t0': integral_time[i+1]})
        options.update({'t1': integral_time[i]})
        z_t0, g_t0, logp_diff_t0,_= odesolve(func,y0=(x, logp_diff_t1, logp_diff_t1, g_t1),options=options)
        L2_value3[0][i] = group_distance(z_t0, data_train[i])#mse(z_t0,noisy_points)
        loss = loss  + L2_value3[0][i]
        ############################################
        x2 = Sampling(args.num_samples, train_time,i,data_train,0.0001,device)#
        x2.requires_grad=True
        logp_diff_t1 = torch.zeros(x2.shape[0], 1).type(torch.float32).to(device)
        g_t1 = torch.zeros_like(x2).type(torch.float32).to(device)
        # loss between each two time points
        options.update({'t0': integral_time[i]})
        options.update({'t1': integral_time[i+1]})
        z_t0, g_t0, logp_diff_t0,_= odesolve(func,y0=(x2, logp_diff_t1, logp_diff_t1, g_t1),options=options)
        L2_value4[0][i] = group_distance(z_t0, data_train[i+1])#mse(z_t0,noisy_points)#mse(data_train[0],z_t0)
        loss = loss  + L2_value4[0][i]
        
    if (itr >1):
        if ((itr % 100 == 0) and (itr<=args.niters-400) and (sigma_now>0.001)):
            sigma_now = sigma_now/2

    return loss, sigma_now, L2_value3, L2_value4
            

def train_model(mse,func,args,data_train,train_time,integral_time,sigma_now,options,device,itr):
    warnings.filterwarnings("ignore")
    loss = 0
    L2_value1 = torch.zeros(1,len(data_train)-1).type(torch.float32).to(device)
    L2_value2 = torch.zeros(1,len(data_train)-1).type(torch.float32).to(device)
    L2_value3 = torch.zeros(1,len(data_train)-1).type(torch.float32).to(device)
    L2_value4 = torch.zeros(1,len(data_train)-1).type(torch.float32).to(device)
    odeint_setp = gcd_list([num * 100 for num in integral_time])/100
    a=0.01
    for i in range(len(train_time)-1): 
        x = Sampling(args.num_samples, train_time,i+1,data_train,a,device)#0.02
        #x = ImportanceSampling(args.num_samples, train_time,i+1,data_train,a,device)#0.02
        x.requires_grad=True
        logp_diff_t1 = torch.zeros(x.shape[0], 1).type(torch.float32).to(device)
        g_t1 = torch.zeros_like(x).type(torch.float32).to(device)
        options.update({'t0': integral_time[i+1]})
        options.update({'t1': integral_time[0]})
        z_t0, g_z, logp_diff_t0, logp_dz = odesolve(func,y0=(x,logp_diff_t1,logp_diff_t1,g_t1),options=options)
        #aa = MultimodalGaussian_density(z_t0, train_time, 0, data_train,sigma_now,device) #normalized density
        aa = MultimodalGaussian_density_sample(z_t0, train_time, 0, data_train,sigma_now,device) #normalized density
        zero_den = (aa < 1e-16).nonzero(as_tuple=True)[0]
        aa[zero_den] = torch.tensor(1e-16).type(torch.float32).to(device)
        logp_x = torch.log(aa)-logp_diff_t0.view(-1)
        #aaa = MultimodalGaussian_density(x, train_time, i+1, data_train,sigma_now,device) * torch.tensor(data_train[i+1].shape[0]/data_train[0].shape[0]) # mass
        aaa = MultimodalGaussian_density_sample(x, train_time, i+1, data_train,sigma_now,device) * torch.tensor(data_train[i+1].shape[0]/data_train[0].shape[0]) # mass
        L2_value1[0][i] = mse(aaa,torch.exp(logp_x.view(-1)))
        loss = loss  + L2_value1[0][i]*1e4 #+ L2_value3[0][i]
        
        # loss between each two time points
        options.update({'t0': integral_time[i+1]})
        options.update({'t1': integral_time[i]})
        z_t0, g_z, logp_diff_t0, logp_dz = odesolve(func,y0=(x,logp_diff_t1,logp_diff_t1,g_t1),options=options)
        
        #aa = MultimodalGaussian_density(z_t0, train_time, i, data_train,sigma_now,device)* torch.tensor(data_train[i].shape[0]/data_train[0].shape[0])
        aa = MultimodalGaussian_density_sample(z_t0, train_time, i, data_train,sigma_now,device)* torch.tensor(data_train[i].shape[0]/data_train[0].shape[0])
        #find zero density
        zero_den = (aa < 1e-16).nonzero(as_tuple=True)[0]
        aa[zero_den] = torch.tensor(1e-16).type(torch.float32).to(device)
        logp_x = torch.log(aa)-logp_diff_t0.view(-1)
        
        L2_value2[0][i] = mse(aaa,torch.exp(logp_x.view(-1))) 
        loss = loss  + L2_value2[0][i]*1e4 #+ L2_value4[0][i]
        
    # compute transport cost efficiency
    transport_cost = partial(trans_loss,func=func,device=device,odeint_setp=odeint_setp,integral_time=integral_time,train_time=train_time,data_train=data_train,sigma_now=0.5)
    x0 = Sampling(args.num_samples,train_time,0,data_train,a,device) 
    #x0 = ImportanceSampling(args.num_samples,train_time,0,data_train,a,device) 
    g_t00 = torch.zeros(x0.shape[0], 1).type(torch.float32).to(device)
    logp_diff_z = torch.zeros_like(x0).type(torch.float32).to(device)
    _,_,loss1,_= odeint(transport_cost,y0=(x0, g_t00, g_t00,logp_diff_z),t = torch.tensor([0, integral_time[-1]]).type(torch.float32).to(device),atol=1e-5,rtol=1e-5,method='midpoint',options = {'step_size': odeint_setp})
    # set inf and nan to 0
    loss11=loss1
    loss11[loss11 == float('-inf')] = 0
    loss11[loss11 == float('inf')] = 0
    loss11[loss11 == 'nan'] = 0
    loss = loss + integral_time[-1]*(loss11[-1].mean(0))#+2*func.d*(et-e0))
    
    if (itr >1):
        if ((itr % 100 == 0)  and (itr<=args.niters-200) and (L2_value1.mean()<=0.0003)):#
            if (sigma_now>0.02):
               sigma_now = sigma_now/2
            a=a/2

    return loss, loss1, sigma_now, L2_value1, L2_value2, L2_value3, L2_value4
            

# plot 3d of inferred trajectory of 20 cells
def plot_3d(func,data_train,train_time,integral_time,args,device):
    viz_samples = 20
    sigma_a = 0.001
    t_list = []
    z_t_samples = []
    z_t_data = []
    v = []
    g = []
    t_list2 = [] 
    odeint_setp = gcd_list([num * 100 for num in integral_time])/100
    integral_time2 = np.arange(integral_time[0], integral_time[-1]+odeint_setp, odeint_setp)
    integral_time2 = np.round_(integral_time2, decimals = 2)
    plot_time = list(reversed(integral_time2))
    sample_time = np.where(np.isin(np.array(plot_time),integral_time))[0]
    sample_time = list(reversed(sample_time))

    with torch.no_grad():
        for i in range(len(integral_time)):

            z_t0 =  data_train[i]
            z_t_data.append(z_t0.cpu().detach().numpy())
            t_list2.append(integral_time[i])
        
        # traj backward
        z_t0 =  Sampling(viz_samples, train_time, len(train_time)-1,data_train,sigma_a,device)
        logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        g0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        v_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device),(z_t0,g0, logp_diff_t0))[0] #True_v(z_t0)
        g_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device),(z_t0,g0, logp_diff_t0))[1]
        v.append(v_t.cpu().detach().numpy())
        g.append(g_t.cpu().detach().numpy())
        z_t_samples.append(z_t0.cpu().detach().numpy())
        t_list.append(plot_time[0])
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': None})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-5})
        options.update({'print_neval': False})
        options.update({'neval_max': 1000000})
        options.update({'safety': None})
        options.update({'t0': integral_time[-1]})
        options.update({'t1': 0})
        options.update({'t_eval':plot_time})
        z_t1,_, logp_diff_t1= odesolve(func,y0=(z_t0,g0, logp_diff_t0),options=options)
        for i in range(len(plot_time)-1):
            v_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t1[i+1], g0, logp_diff_t1))[0] #True_v(z_t0)
            g_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t1[i+1], g0, logp_diff_t1))[1]
            
            z_t_samples.append(z_t1[i+1].cpu().detach().numpy())
            g.append(g_t.cpu().detach().numpy())
            v.append(v_t.cpu().detach().numpy())
            t_list.append(plot_time[i+1])

        aa=5#3
        angle1 = 10#30
        angle2 = 75#30
        widths = 0.2 #arrow width
        fig = plt.figure(figsize=(4*2,3*2), dpi=200)
        plt.tight_layout()
        plt.margins(0, 0)
        v_scale = 5
        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)
        #fig.suptitle(f'{t:.1f}day')
        ax1 = plt.axes(projection ='3d')
        ax1.grid(False)
        ax1.set_xlabel('UMAP1')
        ax1.set_ylabel('UMAP2')
        ax1.set_zlabel('UMAP3')
        ax1.set_xlim(-2,2)
        ax1.set_ylim(-2,2)
        ax1.set_zlim(-2,2)
        ax1.set_xticks([-2,2])
        ax1.set_yticks([-2,2])
        ax1.set_zticks([-2,2])
        ax1.view_init(elev=angle1, azim=angle2)
        #ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))

        ax1.invert_xaxis()
        ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 1, 0.7, 1]))
        line_width = 0.3
        color_wanted = [np.array([250,187,110])/255,
                        np.array([173,219,136])/255,
                        np.array([250,199,179])/255,
                        np.array([238,68,49])/255,
                        np.array([206,223,239])/255,
                        np.array([3,149,198])/255,
                        np.array([180,180,213])/255,
                        np.array([178,143,237])/255]
        for j in range(viz_samples): #individual traj
            for i in range(len(plot_time)-1):
                ax1.plot([z_t_samples[i][j,0],z_t_samples[i+1][j,0]],
                            [z_t_samples[i][j,1],z_t_samples[i+1][j,1]],
                            [z_t_samples[i][j,2],z_t_samples[i+1][j,2]],
                            linewidth=0.5,color ='grey',zorder=2)

        # add inferrred trajecotry
        for i in range(len(sample_time)):
            #ax1.scatter(z_t_samples[sample_time[i]][:,0],z_t_samples[sample_time[i]][:,1],z_t_samples[sample_time[i]][:,2],s=aa*10,linewidth=0, color=color_wanted[i],zorder=3)
            ax1.quiver(z_t_samples[sample_time[i]][:,0],z_t_samples[sample_time[i]][:,1],z_t_samples[sample_time[i]][:,2],
                       v[sample_time[i]][:,0]/v_scale,v[sample_time[i]][:,1]/v_scale,v[sample_time[i]][:,2]/v_scale, color='k',alpha=1,linewidths=widths*2,arrow_length_ratio=0.3,zorder=4)
                
        for i in range(len(integral_time)):
            ax1.scatter(z_t_data[i][:,0],z_t_data[i][:,1],z_t_data[i][:,2],s=aa,linewidth=line_width,alpha = 0.7, facecolors='none', edgecolors=color_wanted[i],label=integral_time[i],zorder=1)

        #plt.savefig(os.path.join(args.save_dir, f"traj_3d.pdf"),format="pdf",pad_inches=0.1, bbox_inches='tight')
        ax1.legend(loc='upper right')
        plt.show()
            
def Jacobian(f, z):
    """Calculates Jacobian df/dz.
    """
    jac = []
    for i in range(f.shape[1]):
        df_dz = torch.autograd.grad(f[:, i], z, torch.ones_like(f[:, i]),retain_graph=True, create_graph=True)[0].view(z.shape[0], -1)
        jac.append(torch.unsqueeze(df_dz, 1))
    jac = torch.cat(jac, 1)
    return jac
## Calculate growth-related top 10 promote and inhibite genes
def Growth_ave_top(viz_samples,z_t0,time_pt,device,max_use,min_use,mean_use,gene_name,filepath,title,func,pca_model='nan',model='nan'):
    dim = len(gene_name)
    dim2 = z_t0.shape[1]
    gg = np.zeros((1,dim))
    g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
    p_z = torch.zeros(1, dim2).type(torch.float32).to(device)
    max_min = torch.tensor(max_use-min_use).type(torch.float32).to(device)
    for i in range(viz_samples):
        x_t = z_t0[i,:].reshape([1,dim2])
        g_xt = func(torch.tensor(time_pt).type(torch.float32).to(device),(x_t,g_xt0, g_xt0,p_z))[1]
        if pca_model=='nan':
            x_gene = model.get_generative(x_t*max_min/3+torch.tensor(mean_use).type(torch.float32).to(device))
            x_latent = model.get_latent_representation(x_gene)
        else:
            x_t2 = ((x_t + 2) / (max_use - min_use))* 0.25 + min_use#正规化到[-2,2]
            x_gene = pca_model.inverse_transform(x_t2)
            x_latent = pca_model.transform(x_gene)
            x_latent = (x_latent - min_use) / (max_use - min_use) * 4 - 2#正规化到[-2,2]
        
        gg1=torch.autograd.grad(g_xt, x_t, torch.ones_like(g_xt),retain_graph=True, create_graph=True)[0].view(x_t.shape[0], -1).reshape(1,dim2).detach().cpu().numpy()
        jac2 = Jacobian(x_latent, x_gene).reshape(dim2,x_gene.shape[1]).detach().cpu().numpy()
        gg = gg + np.matmul(gg1,jac2)
    gg = gg/viz_samples
    gg = gg.flatten()
    top_10_indices = np.argsort(gg)[-10:][::-1]   # top 10
    bottom_10_indices = np.argsort(gg)[:10][::-1]   # bottom 10
    top_10_values = gg[top_10_indices]
    bottom_10_values = gg[bottom_10_indices]
    # Create dataframes for the heatmaps
    top_10_df = pd.DataFrame(data=top_10_values, index=[gene_name[i] for i in top_10_indices.flatten()], columns=['Top 10'])
    bottom_10_df = pd.DataFrame(data=bottom_10_values, index=[gene_name[i] for i in bottom_10_indices.flatten()], columns=['Bottom 10'])
    # Define the normalization for the color scale
    divnorm1 = colors.TwoSlopeNorm(vmin=np.min(gg), vcenter=0., vmax=np.max(gg))
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(4, 4), dpi=200, sharex=False, sharey=False)
    cbar_ax = fig.add_axes([.91, .3, .02, .4])
    # Plot top 10 heatmap
    sns.heatmap(top_10_df, ax=axes[0], cmap="coolwarm", norm=divnorm1, xticklabels=[], yticklabels=True, annot=False, cbar=False)
    axes[0].set_title('Top 10')
    # Plot bottom 10 heatmap
    sns.heatmap(bottom_10_df, ax=axes[1], cmap="coolwarm", norm=divnorm1, xticklabels=[], yticklabels=True, annot=False, cbar_ax=cbar_ax)
    axes[1].set_title('Bottom 10')
    plt.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig(filepath+title+".pdf",format="pdf",pad_inches=0.2, bbox_inches='tight')
    plt.show()

def Jac_ave_sub(viz_samples,z_t0,time_pt, device,max_use,min_use,mean_use,gene_index,func,pca_model='nan',model='nan'):
    dim = len(gene_index)
    dim2 = z_t0.shape[1]
    jac = np.zeros((dim,dim))
    g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
    p_z = torch.zeros(1, dim2).type(torch.float32).to(device)
    max_min = torch.tensor(max_use-min_use).type(torch.float32).to(device)
    for i in range(viz_samples):
        x_t = z_t0[i,:].reshape([1,dim2])
        v_xt = func(torch.tensor(time_pt).type(torch.float32).to(device),(x_t,g_xt0, g_xt0,p_z))[0]
        if pca_model=='nan':
            v_t_gene2 = model.get_generative((x_t+v_xt)*max_min/3+torch.tensor(mean_use).type(torch.float32).to(device))
            x_gene = model.get_generative(x_t*max_min/3+torch.tensor(mean_use).type(torch.float32).to(device))
            x_latent = model.get_latent_representation(x_gene)
        else:
            x_t2 = ((x_t+v_xt + 2) / (max_use - min_use))* 0.25 + min_use#normalize to [-2,2]
            v_t_gene2 = pca_model.inverse_transform(x_t2)
            x_t3 = ((x_t + 2) / (max_use - min_use))* 0.25 + min_use#normalize to [-2,2]
            x_gene = pca_model.inverse_transform(x_t3)
            x_latent = pca_model.transform(x_gene)
            x_latent = (x_latent - min_use) / (max_use - min_use) * 4 - 2#normalize to [-2,2]
        v_t_gene = v_t_gene2 - x_gene
        jac1 = Jacobian(v_t_gene[:,gene_index], x_t).reshape(dim,dim2).detach().cpu().numpy()
        jac2 = Jacobian(x_latent, x_gene).reshape(dim2,x_gene.shape[1]).detach().cpu().numpy()
        jac2 = jac2[:,gene_index]
        jac = jac + np.matmul(jac1,jac2)
    jac = jac/viz_samples
    return jac
    
# plot avergae jac of v of cells (z_t) at time (time_pt)
def plot_jac_v(func,z_t,time_pt,title,gene_list,args,device):
    g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
    logp_diff_xt0 = g_xt0
    # compute the mean of jacobian of v within cells z_t at time (time_pt)
    dim = z_t.shape[1]
    p_z = torch.zeros(1, dim).type(torch.float32).to(device)
    jac = np.zeros((dim,dim))
    for i in range(z_t.shape[0]):
        x_t = z_t[i,:].reshape([1,dim])
        v_xt = func(torch.tensor(time_pt).type(torch.float32).to(device),(x_t,g_xt0, logp_diff_xt0,p_z))[0]
        jac = jac+Jacobian(v_xt, x_t).reshape(dim,dim).detach().cpu().numpy()
    jac = jac/z_t.shape[0]
    print(jac)
    print(jac.shape)
    fig = plt.figure(figsize=(5, 4), dpi=200)
    ax = fig.add_subplot(111)
    plt.tight_layout()
    ax.set_title('Jacobian of velocity')
    sns.heatmap(jac,cmap="coolwarm",annot=True,fmt=".2f", cbar_kws={"shrink": .8}, annot_kws={"fontsize": 10})#,xticklabels=gene_list,yticklabels=gene_list
    ax.set_yticklabels(gene_list, rotation=0) 
    ax.set_xticklabels(gene_list, minor=False)  
    ax.set_yticklabels(gene_list, minor=False) 
    #plt.savefig(os.path.join(args.save_dir, title),format="pdf",pad_inches=0.2, bbox_inches='tight')
    plt.show()

def plot_jac_v2(JAC,plot_time,index1,gene_name,list_name,filepath, k=0):
    
    jac = JAC
    subfigure = len(plot_time)
    divnorm1=colors.TwoSlopeNorm(vmin=np.min(np.min(jac)), vcenter=0., vmax=np.max(np.max(jac)))
    fig, axes = plt.subplots(1,subfigure,figsize=(3*subfigure, 2.5), dpi=200, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .005, .4])
    gene_list1 = [gene_name[i] for i in index1]
    time_all = ['0d','2d','6d','10d']
    #c_type = ['E','pEMT','MET','M']
    for i, ax in enumerate(axes.flat):
        df = pd.DataFrame(data = jac[i], 
                            index = gene_list1, 
                            columns = gene_list1)
        ax.set_title('Jacobian of velocity at '+ time_all[i])
        #ax.set_title('Jac of vel of '+c_type[k]+' at '+ time_all[i])
        sns.heatmap( df,ax=ax, cmap="coolwarm", norm=divnorm1, cbar=i == 0,cbar_ax=None if i else cbar_ax)
        #ax.set_xticklabels(gene_list1, rotation=90)
        #ax.set_yticklabels(gene_list1)

    plt.savefig(filepath+list_name+".pdf",format="pdf",pad_inches=0.2, bbox_inches='tight')
                   
def plot_jac_v_all(func,data_train,title,gene_list,filepath,args,device):
    JAC=[]
    integral_time=args.timepoints
    for j in range(len(data_train)):
        g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
        # compute the mean of jacobian of v within cells z_t at time (time_pt)
        dim = data_train[j].shape[1]
        logp_diff_xt0 = torch.zeros(1, dim).type(torch.float32).to(device)
        jac = np.zeros((dim,dim))
        for i in range(data_train[j].shape[0]):
            x_t = data_train[j][i,:].reshape([1,dim])
            v_xt = func(torch.tensor(integral_time[j]).type(torch.float32).to(device),(x_t,g_xt0,g_xt0, logp_diff_xt0))[0]
            jac = jac+Jacobian(v_xt, x_t).reshape(dim,dim).detach().cpu().numpy()
        jac = jac/data_train[j].shape[0]
        JAC.append(jac)
    
    subfigure = len(integral_time)
    divnorm1=colors.TwoSlopeNorm(vmin=np.min(np.min(JAC)), vcenter=0., vmax=np.max(np.max(JAC)))
    fig, axes = plt.subplots(1,subfigure,figsize=(3*subfigure, 2.5), dpi=200, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .005, .4])
    #time_all# = ['0d','8h','1d','3d','7d']
    for i, ax in enumerate(axes.flat):
        df = pd.DataFrame(data = JAC[i], 
                            index = gene_list, 
                            columns = gene_list)
        ax.set_title('Jacobian of velocity at {:.0f}'.format(integral_time[i]))
        sns.heatmap(df,ax=ax, cmap="coolwarm", norm=divnorm1, cbar=i == 0, annot=True,cbar_ax=None if i else cbar_ax)
        #ax.set_xticklabels(gene_list1, rotation=90)
        ax.set_yticklabels(gene_list, rotation=90)
    plt.savefig(filepath+title+"_"+args.dataset+".pdf",format="pdf",pad_inches=0.2, bbox_inches='tight')

def plot_grn(JAC,plot_time,index1,gene_name,list_name,filepath,threshold= 0.05,k=0):
    colors=['#d62728', '#d62728',
            '#2ca02c','#2ca02c','#2ca02c','#2ca02c']
    sacle_link = 3
    jac_all = JAC
    jac_all = jac_all/np.max(jac_all)
    dim = jac_all[0].shape[0]
    g1_x = np.zeros((dim,))
    g1_y = np.arange(dim)[::-1]
    g2_x = np.ones((dim,))*dim
    g2_y = g1_y
    dot_size = 200
    aa=0.4
    subfigure = len(plot_time)
    fig = plt.figure(figsize=(3*subfigure, 3), dpi=200)
    plt.tight_layout()
    plt.axis('off')
    plt.margins(0.2, 0.2)
    gene_list1 = [gene_name[i] for i in index1]
    time_all = ['0d','2d','6d','10d']
    c_type = ['E','pEMT','MET','M']
    for ii in range(subfigure):
        ax1 = fig.add_subplot(1, subfigure, ii+1)
        jac = jac_all[ii]
        ax1.set_title('GRN at '+ time_all[ii])
        #ax1.set_title('GRN of '+c_type[k]+' at '+ time_all[ii])
        ax1.axis('off')
        ax1.margins(0.2, 0.2)

        for i in range(dim):
            ax1.scatter(g1_x[i],g1_y[i],s=dot_size*2,linewidth=0,zorder=1,alpha=0.8,color=colors[i])
            ax1.scatter(g2_x[i],g2_y[i],s=dot_size*2,linewidth=0,zorder=1,alpha=0.8,color=colors[i])

        for i in range(dim):
            for j in range (dim):
                link = jac[j,i]
                if link>threshold:
                    ax1.annotate("",xytext=(g1_x[i]+aa, g1_y[i]),xy=(g2_x[j]-aa, g2_y[j]),
                                arrowprops=dict(arrowstyle="->",mutation_scale=10,linewidth=link*sacle_link,color=colors[i]),alpha=0.8,zorder=2)
                if link<-threshold:
                    ax1.annotate("",xytext=(g1_x[i]+aa, g1_y[i]),xy=(g2_x[j]-aa, g2_y[j]),
                                arrowprops=dict(arrowstyle="-[",linewidth=-link*sacle_link,mutation_scale=3,color=colors[i]),alpha=0.8,zorder=2)
                if j==0:
                    ax1.text(g1_x[i], g1_y[i], gene_list1[i], ha='center', va='center', fontsize=8,zorder=3)
                    
    plt.savefig(filepath+list_name+".png",format="png",pad_inches=0.2, bbox_inches='tight',dpi=300) 
        
def plot_grn_all(func,data_train,title,gene_list,filepath,args,device,threshold):
    colors = [np.array([10,211,12])/255,#np.array([145,211,192])/255
              np.array([9,147,250])/255,
              np.array([174,32,18])/255,
                    np.array([188,62,3])/255,
                    np.array([204,102,2])/255,
                    np.array([238,155,0])/255,
                    np.array([235,215,165])/255,
                    ]
    JAC=[]
    integral_time=args.timepoints
    for j in range(len(data_train)):
        g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
        # compute the mean of jacobian of v within cells z_t at time (time_pt)
        dim = data_train[j].shape[1]
        logp_diff_xt0 = torch.zeros(1, dim).type(torch.float32).to(device)
        jac = np.zeros((dim,dim))
        for i in range(data_train[j].shape[0]):
            x_t = data_train[j][i,:].reshape([1,dim])
            v_xt = func(torch.tensor(integral_time[j]).type(torch.float32).to(device),(x_t,g_xt0,g_xt0, logp_diff_xt0))[0]
            jac = jac+Jacobian(v_xt, x_t).reshape(dim,dim).detach().cpu().numpy()
        jac = jac/data_train[j].shape[0]
        JAC.append(jac)
    
    sacle_link = 4
    JAC = JAC/np.max(JAC)
    dim = JAC[0].shape[0]
    g1_x = np.zeros((dim,))
    g1_y = np.arange(dim)[::-1]
    g2_x = np.ones((dim,))*dim
    g2_y = g1_y
    dot_size = 200
    aa=0.4
    subfigure = len(integral_time)
    fig = plt.figure(figsize=(3*subfigure, 2.5), dpi=200)
    plt.tight_layout()
    plt.axis('off')
    plt.margins(0.2, 0.2)
    for ii in range(subfigure):
        ax1 = fig.add_subplot(1, subfigure, ii+1)
        jac = JAC[ii]
        ax1.set_title('GRN at {:.0f}'.format(integral_time[ii]))
        ax1.axis('off')
        ax1.margins(0.2, 0.2)
        for i in range(dim):
            ax1.scatter(g1_x[i],g1_y[i],s=dot_size*2,linewidth=0,zorder=1,alpha=0.8,color=colors[i])
            ax1.scatter(g2_x[i],g2_y[i],s=dot_size*2,linewidth=0,zorder=1,alpha=0.8,color=colors[i])
        for i in range(dim):
            for j in range (dim):
                link = jac[j,i]
                if link>threshold:
                    ax1.annotate("",xytext=(g1_x[i]+aa, g1_y[i]),xy=(g2_x[j]-aa, g2_y[j]),
                                arrowprops=dict(arrowstyle="->",mutation_scale=10,linewidth=link*sacle_link,color=colors[i]),alpha=0.8,zorder=2)
                if link<-threshold:
                    ax1.annotate("",xytext=(g1_x[i]+aa, g1_y[i]),xy=(g2_x[j]-aa, g2_y[j]),
                                arrowprops=dict(arrowstyle="-[",linewidth=-link*sacle_link,mutation_scale=3,color=colors[i]),alpha=0.8,zorder=2)
                if j==0:
                    ax1.text(g1_x[i], g1_y[i], gene_list[i], ha='center', va='center', fontsize=7,zorder=3)
    
    plt.savefig(filepath+title+"_"+args.dataset+".pdf",format="pdf",pad_inches=0.2, bbox_inches='tight')                
               
# plot avergae gradients of g of cells (z_t) at time (time_pt)
def plot_grad_g(func,z_t,time_pt,title,gene_list,args,device,filepath):
    g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
    logp_diff_xt0 = g_xt0
    dim = z_t.shape[1]
    p_z = torch.zeros(1, dim).type(torch.float32).to(device)
    gg = np.zeros((dim,1))
    for i in range(z_t.shape[0]):
        x_t = z_t[i,:].reshape([1,dim])
        g_xt = func(torch.tensor(time_pt).type(torch.float32).to(device),(x_t,g_xt0, logp_diff_xt0,p_z))[1]
        gg = gg+torch.autograd.grad(g_xt, x_t, torch.ones_like(g_xt),retain_graph=True, create_graph=True)[0].view(x_t.shape[0], -1).reshape(dim,1).detach().cpu().numpy()
    gg = gg/z_t.shape[0]
    print(gg)
    fig= plt.figure(figsize=(1.5, 4), dpi=200)
    ax = fig.add_subplot(111)
    plt.tight_layout()
    plt.margins(0, 0)
    ax.set_title('Gradient of growth')
    sns.heatmap(gg,cmap="coolwarm",xticklabels=[],yticklabels=gene_list,annot=True)#,annot_kws={"size": 8}
    ax.set_yticklabels(gene_list, rotation=0)
    ax.set_yticklabels(gene_list, minor=False)  # 设置 y 轴主刻度标签
    plt.savefig(filepath+title+"_"+args.dataset+".pdf",format="pdf",pad_inches=0.2, bbox_inches='tight')
    plt.show()

def plot_grad_g_all(func,data_train,title,gene_list,filepath,args,device):#func,data_train,title,gene_list,filepath,args,device
    GG=[]
    integral_time=args.timepoints
    for j in range(len(data_train)):
        g_xt0 = torch.zeros(1, 1).type(torch.float32).to(device)
        logp_diff_xt0 = g_xt0
        dim = data_train[j].shape[1]
        p_z = torch.zeros(1, dim).type(torch.float32).to(device)
        gg = np.zeros((dim,1))
        for i in range(data_train[j].shape[0]):
            x_t = data_train[j][i,:].reshape([1,dim])
            g_xt = func(torch.tensor(integral_time[j]).type(torch.float32).to(device),(x_t,g_xt0, logp_diff_xt0,p_z))[1]
            gg = gg+torch.autograd.grad(g_xt, x_t, torch.ones_like(g_xt),retain_graph=True, create_graph=True)[0].view(x_t.shape[0], -1).reshape(dim,1).detach().cpu().numpy()
        gg = gg/data_train[j].shape[0]
        GG.append(gg)
    
    subfigure = len(integral_time)
    divnorm1=colors.TwoSlopeNorm(vmin=np.min(np.min(GG)), vcenter=0., vmax=np.max(np.max(GG)))
    fig, axes = plt.subplots(1,subfigure,figsize=(1*subfigure, 4), dpi=200, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .02, .4])
    #time_all# = ['0d','8h','1d','3d','7d']
    for i, ax in enumerate(axes.flat):
        df = pd.DataFrame(data = GG[i], 
                            index = gene_list, 
                            )#columns = []#gene_list
        ax.set_title('Time: {:.0f}'.format(integral_time[i]))#,fontsize=6
        sns.heatmap(df,ax=ax, cmap="coolwarm", norm=divnorm1,xticklabels=[], cbar=i == 0, annot=True,cbar_ax=None if i else cbar_ax)
        ax.set_yticklabels(gene_list, rotation=90)
        ax.set_yticklabels(gene_list, minor=False)  # 设置 y 轴主刻度标签
        
    plt.savefig(filepath+title+"_"+args.dataset+".pdf",format="pdf",pad_inches=0.2, bbox_inches='tight')
    plt.show()


def plot_3d_landscape(func,data_train,train_time,integral_time,args,device):
    m=16
    odeint_setp = gcd_list([num * 100 for num in integral_time])/1000
    integral_time2 = np.arange(integral_time[0], integral_time[-1]+odeint_setp, odeint_setp)
    integral_time2 = np.round_(integral_time2, decimals = 2)
    plot_time = list(integral_time2)
    sigma_now=0.01#####func.d
    with torch.no_grad():
        ###################################################################################
        all_data = np.vstack([data.cpu().numpy() for data in data_train])
        min_values = np.min(all_data, axis=0)
        max_values = np.max(all_data, axis=0)
        ###########################################################
        fig = plt.figure(figsize=(12, 15), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        time_pt_list = integral_time
        num_points = int(100)  # 
        m = 16  ##maxinum of energy
        v_scale = 1  
        X, Y = np.meshgrid(np.linspace(min_values[0] - 1, max_values[0] + 1, num_points),
                   np.linspace(min_values[1] - 1, max_values[1] + 1, num_points))
        z_t0 = np.column_stack((X.ravel(), Y.ravel()))
        z_t0=torch.tensor(z_t0).type(torch.float32).to(device)
        ###################################################
        x1 = np.linspace(min_values[0]-1, max_values[0]+1, 10)
        y1 = np.linspace(min_values[1]-1, max_values[1]+1, 10)
        X1, Y1 = np.meshgrid(x1, y1)
        color_wanted = ['#d62728', '#2ca02c', '#ff7f0e',
                        '#e377c2', '#17becf', '#bcbd22', 
                        '#1f77b4', '#9467bd', '#8c564b', 
                        '#7f7f7f','#2f7f2e']
        
        # Randomly sample 'sample_size' indices from the total number of samples
        sample_size=100
        size=[len(data_train[i]) for i in range(len(integral_time))]
        num_all= min(size)
        sample_indices = torch.randint(0, num_all, (sample_size,))#[sample_indices]
        unique_data_types = ['E','M','MET','pEMT']
        color_palette = sns.color_palette(color_wanted, len(unique_data_types)) 
        palette = {cell_type: color for cell_type, color in zip(unique_data_types, color_palette)}
        for idx,time_pt in enumerate(time_pt_list):
            ###################### 
            start_time,end_time=0,1
            for i in range(len(integral_time)-1):
              if integral_time[i]<=time_pt<integral_time[i+1]:
                  start_time=i
                  end_time=i+1
                  #k=i+1
                  break
              elif time_pt==integral_time[-1]:
                  start_time=len(integral_time)-2
                  end_time=len(integral_time)-1
                  #k=len(integral_time)
                  break
              elif time_pt>integral_time[-1]:
                  print("The time t should less than {}".format(integral_time[-1]))
                  break
            #####################  estimate the probability
            prob_density1 = MultimodalGaussian_density(z_t0, train_time, start_time, data_train,sigma_now,device)
            prob_density2 = MultimodalGaussian_density(z_t0, train_time, end_time, data_train,sigma_now,device)
            
            p_i=(integral_time[end_time]-time_pt)/(integral_time[end_time]-integral_time[start_time])
            p_t=p_i*prob_density1 + (1-p_i)*prob_density2 
            ################################################################################
            z_t1 = np.column_stack((X1.ravel(), Y1.ravel()))
            z_t1=torch.tensor(z_t1).type(torch.float32).to(device)#.type(torch.float32)
            logp_diff_t0 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)
            g0 = torch.zeros(z_t1.shape).type(torch.float32).to(device)
            s_t = func(torch.tensor(time_pt).type(torch.float32).to(device),(z_t1,logp_diff_t0,logp_diff_t0,g0))[3]
            ############################################################################
            L_p = -np.log(np.array(p_t.cpu().detach().numpy()).reshape(num_points, num_points))
            L_p[L_p > m] = m
            ax.contourf(X, Y, L_p, offset=time_pt_list[idx], levels=50, cmap='rainbow', alpha=0.7,zorder=0)#time_pt
            z_t1=z_t1.cpu().detach().numpy()
            ax.quiver(z_t1[:, 0], z_t1[:, 1], np.full_like(z_t1[:, 0], time_pt_list[idx]),  
                s_t[:, 0] / v_scale, s_t[:, 1] / v_scale, np.zeros_like(z_t1[:, 0]),  
                color='k', length=0.2, arrow_length_ratio=0.1,normalize=True,alpha=1,linewidths=1.0,zorder=3)  
            
        ax.set_xlim(int(min_values[0]) - 1.5, int(max_values[0]) + 1.5)
        ax.set_ylim(int(min_values[1]) - 1.5, int(max_values[1]) + 1.5)
        ax.set_zlim(0, max(time_pt_list) + 0.1)
        ax.set_zlim(ax.get_zlim()[::-1])  #
        ax.set_xlabel('X1', fontsize=14)
        ax.set_ylabel('X2', fontsize=14)
        ax.set_zlabel('Time', fontsize=14)
        ax.set_title("3D Landscape Over Time", fontsize=16)
        ax.view_init(elev=10)  
        ax.axis('off')
        plt.savefig(os.path.join(args.save_dir, "Landscape_"+args.dataset+".png"),format="png",pad_inches=0.1, bbox_inches='tight',dpi=300)
        plt.show()


# plot 2d of inferred trajectory of 20 cells
def plot_2d(func,data_train,train_time,integral_time,args,device):
    viz_samples = 300
    sigma_a = 0.001
    t_list = []
    z_t_samples = []
    z_t_data = []
    z_t_data2 = []
    v = []
    g = []
    gg = []
    s = []
    t_list2 = [] 
    odeint_setp = gcd_list([num * 100 for num in integral_time])/100
    integral_time2 = np.arange(integral_time[0], integral_time[-1]+odeint_setp, odeint_setp)
    integral_time2 = np.round_(integral_time2, decimals = 2)
    plot_time = list(reversed(integral_time2))
    sample_time = np.where(np.isin(np.array(plot_time),integral_time))[0]
    sample_time = list(reversed(sample_time))
    
    with torch.no_grad():
        for i in range(len(integral_time)):

            z_t0 =  data_train[i]
            z_t_data.append(z_t0.cpu().detach().numpy())
            t_list2.append(integral_time[i])
        
        all_data = np.vstack([data.cpu().numpy() for data in data_train])
        min_values = np.min(all_data, axis=0)
        max_values = np.max(all_data, axis=0)
        
        #z_t0 =  Sampling(viz_samples, train_time, len(train_time)-1,data_train,sigma_a,device)
        z_t0 =  ImportanceSampling(viz_samples, train_time, len(train_time)-1,data_train,sigma_a,device)
        logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        g0 = torch.zeros(z_t0.shape).type(torch.float32).to(device)
        v_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device),(z_t0,logp_diff_t0,logp_diff_t0,g0))[0]-func.d*func(torch.tensor(integral_time[-1]).type(torch.float32).to(device),(z_t0,logp_diff_t0,logp_diff_t0,g0))[3]#True_v(z_t0)
        s_t = func(torch.tensor(integral_time[-1]).type(torch.float32).to(device),(z_t0,logp_diff_t0,logp_diff_t0,g0))[3]
        for i in range(len(integral_time)):
            viz_data =  Sampling(viz_samples, train_time, i,data_train,sigma_a,device)
            z_t_data2.append(viz_data.cpu().detach().numpy())
            g_t = func(torch.tensor(integral_time[i]).type(torch.float32).to(device),(viz_data,logp_diff_t0,logp_diff_t0,g0))[1]
            gg_t=torch.autograd.grad(g_t, viz_data, torch.ones_like(g_t), create_graph=True)[0].contiguous().contiguous()
            g.append(g_t.cpu().detach().numpy())
            gg.append(gg_t.cpu().detach().numpy())
            
        v.append(v_t.cpu().detach().numpy())
        s.append(s_t.cpu().detach().numpy())
        z_t_samples.append(z_t0.cpu().detach().numpy())
        t_list.append(plot_time[0])
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': None})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-5})
        options.update({'print_neval': False})
        options.update({'neval_max': 1000000})
        options.update({'safety': None})
        options.update({'t0': integral_time[-1]})
        options.update({'t1': 0})
        options.update({'t_eval':plot_time})
        z_t1,_, logp_diff_t1,_= odesolve(func,y0=(z_t0,logp_diff_t0,logp_diff_t0,g0),options=options)
        for i in range(len(plot_time)-1):
            v_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t1[i+1],logp_diff_t0,logp_diff_t1,g0))[0]-func.d*func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t1[i+1],logp_diff_t0,logp_diff_t1,g0))[3] #True_v(z_t0)
            s_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t1[i+1],logp_diff_t0,logp_diff_t1,g0))[3]
            z_t_samples.append(z_t1[i+1].cpu().detach().numpy())
            v.append(v_t.cpu().detach().numpy())
            s.append(s_t.cpu().detach().numpy())
            t_list.append(plot_time[i+1])

        aa=50#3
        widths = 0.2 #arrow width
        ############################################ plot cell velocity
        fig = plt.figure(figsize=(4*2,3*2), dpi=200)
        plt.tight_layout()
        plt.margins(0, 0)
        v_scale = 5
        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)
        ax1 = plt.axes()
        ax1.grid(False)
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_xlim(min_values[0]-0.5,max_values[0]+0.5)
        ax1.set_ylim(min_values[1]-0.5,max_values[1]+0.5)
        ax1.set_xticks([min_values[0]-0.5,max_values[0]+0.5])
        ax1.set_yticks([min_values[1]-0.5,max_values[1]+0.5])
        plt.title('The estimated cell velocity',fontsize=30)
        line_width = 0.3
        color_wanted = [np.array([250,187,110])/255,
                        np.array([173,219,136])/255,
                        np.array([250,199,179])/255,
                        np.array([238,68,49])/255,
                        np.array([206,223,239])/255,
                        np.array([3,149,198])/255,
                        np.array([180,180,213])/255,
                        np.array([178,143,237])/255]
        
        for i in range(len(integral_time)):
            ax1.scatter(z_t_data[i][:,0],z_t_data[i][:,1],#z_t_data[i][:,2],
                        linewidth=line_width,alpha = 0.7, facecolors='none', #s=aa,
                        edgecolors=color_wanted[i],label=integral_time[i])#,zorder=1)

        for j in range(viz_samples): #individual traj
            for i in range(len(plot_time)-1):
                ax1.plot([z_t_samples[i][j,0],z_t_samples[i+1][j,0]],
                            [z_t_samples[i][j,1],z_t_samples[i+1][j,1]],
                            #[z_t_samples[i][j,2],z_t_samples[i+1][j,2]],
                            linewidth=0.5,color ='grey',zorder=1)
                
        # add inferrred trajecotry
        for i in range(len(sample_time)):
            #ax1.scatter(z_t_samples[sample_time[i]][:,0],z_t_samples[sample_time[i]][:,1],z_t_samples[sample_time[i]][:,2],s=aa*10,linewidth=0, color=color_wanted[i],zorder=3)
            ax1.quiver(z_t_samples[sample_time[i]][:,0],z_t_samples[sample_time[i]][:,1],#z_t_samples[sample_time[i]][:,2],
                       v[sample_time[i]][:,0]/v_scale,v[sample_time[i]][:,1]/v_scale,#v[sample_time[i]][:,2]/v_scale,
                       color='k',alpha=1,linewidths=widths*2)#,arrow_length_ratio=0.3,zorder=4)

        for i in range(len(sample_time)):
            ax1.scatter(z_t_samples[sample_time[i]][:,0],z_t_samples[sample_time[i]][:,1],#z_t_data[i][:,2],
                        linewidth=line_width,alpha = 0.7, facecolors=color_wanted[i], s=aa,
                        edgecolors=color_wanted[i],label=integral_time[i])#,zorder=1)
        
        #plt.savefig(os.path.join(args.save_dir, "v_traj_2d_"+args.dataset+"_2.pdf"),format="pdf",pad_inches=0.1, bbox_inches='tight')
        plt.show()
        ############################################ plot trajectory
        fig = plt.figure(figsize=(4*2,3*2), dpi=200)
        plt.tight_layout()
        plt.margins(0, 0)
        v_scale = 5
        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)
        ax1 = plt.axes()
        ax1.grid(False)
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_xlim(min_values[0]-0.5,max_values[0]+0.5)
        ax1.set_ylim(min_values[1]-0.5,max_values[1]+0.5)
        ax1.set_xticks([min_values[0]-0.5,max_values[0]+0.5])
        ax1.set_yticks([min_values[1]-0.5,max_values[1]+0.5])
        plt.title('The estimated cell velocity',fontsize=30)
        line_width = 0.3
        color_wanted = [np.array([250,187,110])/255,
                        np.array([173,219,136])/255,
                        np.array([250,199,179])/255,
                        np.array([238,68,49])/255,
                        np.array([206,223,239])/255,
                        np.array([3,149,198])/255,
                        np.array([180,180,213])/255,
                        np.array([178,143,237])/255]
        
        for i in range(len(integral_time)):
            ax1.scatter(z_t_data[i][:,0],z_t_data[i][:,1],#z_t_data[i][:,2],
                        linewidth=line_width,alpha = 0.7, facecolors='none', #s=aa,
                        edgecolors=color_wanted[i],label=integral_time[i])#,zorder=1)

        for j in range(viz_samples): #individual traj
            for i in range(len(plot_time)-1):
                ax1.plot([z_t_samples[i][j,0],z_t_samples[i+1][j,0]],
                            [z_t_samples[i][j,1],z_t_samples[i+1][j,1]],
                            #[z_t_samples[i][j,2],z_t_samples[i+1][j,2]],
                            linewidth=0.5,color ='grey',zorder=2)
    
        for i in range(len(sample_time)):
            ax1.scatter(z_t_samples[sample_time[i]][:,0],z_t_samples[sample_time[i]][:,1],#z_t_data[i][:,2],
                        linewidth=line_width,alpha = 0.7, facecolors=color_wanted[i], s=aa,
                        edgecolors=color_wanted[i],label=integral_time[i])#,zorder=1)
        
        #plt.savefig(os.path.join(args.save_dir, "traj_2d_"+args.dataset+"_2.pdf"),format="pdf",pad_inches=0.1, bbox_inches='tight'
        plt.show()
        ############################################ plot growth gradient
        fig = plt.figure(figsize=(4*2,3*2), dpi=200)
        plt.tight_layout()
        plt.margins(0, 0)
        v_scale = 5
        plt.tight_layout()
        plt.axis('off')
        plt.margins(0, 0)
        ax1 = plt.axes()
        ax1.grid(False)
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_xlim(min_values[0]-0.5,max_values[0]+0.5)
        ax1.set_ylim(min_values[1]-0.5,max_values[1]+0.5)
        ax1.set_xticks([min_values[0]-0.5,max_values[0]+0.5])
        ax1.set_yticks([min_values[1]-0.5,max_values[1]+0.5])
        plt.title('The growth gradient of cell',fontsize=30)
        for i in range(len(integral_time)):
            scatter=ax1.scatter(z_t_data2[i][:,0],z_t_data2[i][:,1],#z_t_data[i][:,2],
                        linewidth=line_width,alpha = 0.7, facecolors='none', s=100,
                        c=g[i],label=integral_time[i])#,zorder=1),c=g[i],edgecolors=color_wanted[i]
        # add inferrred trajecotry
        for i in range(len(integral_time)):
            #ax1.scatter(z_t_samples[sample_time[i]][:,0],z_t_samples[sample_time[i]][:,1],z_t_samples[sample_time[i]][:,2],s=aa*10,linewidth=0, color=color_wanted[i],zorder=3)
            ax1.quiver(z_t_data2[i][:,0],z_t_data2[i][:,1],#z_t_samples[sample_time[i]][:,2],
                       gg[i][:,0]/v_scale,gg[i][:,1]/v_scale,#v[sample_time[i]][:,2]/v_scale,
                       color='k',alpha=1,linewidths=widths*2)#,arrow_length_ratio=0.3,zorder=4)

        plt.colorbar(scatter)
        #plt.savefig(os.path.join(args.save_dir, "growth_gradient_"+args.dataset+"_2.pdf"),format="pdf",pad_inches=0.1, bbox_inches='tight')
        #plt.savefig(os.path.join(args.save_dir, "growth_gradient_"+args.dataset+"_2.png"),format="png",pad_inches=0.1, bbox_inches='tight',dpi=300)
        plt.show()
        ############################################ plot growth
        fig = plt.figure(figsize=(4*2,3*2), dpi=200)
        v_scale = 5
        ax1 = plt.axes()
        ax1.grid(False)
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_xlim(min_values[0]-0.5,max_values[0]+0.5)
        ax1.set_ylim(min_values[1]-0.5,max_values[1]+0.5)
        ax1.set_xticks([min_values[0]-0.5,max_values[0]+0.5])
        ax1.set_yticks([min_values[1]-0.5,max_values[1]+0.5])
        for i in range(len(integral_time)):
            scatter=ax1.scatter(z_t_data2[i][:,0],z_t_data2[i][:,1],#z_t_data[i][:,2],
                        linewidth=line_width,alpha = 0.7, facecolors='none', s=250,
                        c=g[i],label=integral_time[i])#,zorder=1),c=g[i],edgecolors=color_wanted[i]

        plt.colorbar(scatter)
        #plt.savefig(os.path.join(args.save_dir, "growth_"+args.dataset+"_3.pdf"),format="pdf",pad_inches=0.1, bbox_inches='tight')
        #plt.savefig(os.path.join(args.save_dir, "growth_"+args.dataset+"_3.png"),format="png",pad_inches=0.1, bbox_inches='tight',dpi=300)
        plt.show()
        

# plot 2d of inferred trajectory of 20 cells
def plot_2d_v(func,data_train,train_time,integral_time,args,device):
    viz_samples = 50
    sigma_a = 0.001
    z_t_data = []
    z_t_data2 = []
    v = []
    t_list2 = [] 
    odeint_setp = gcd_list([num * 100 for num in integral_time])/100
    integral_time2 = np.arange(integral_time[0], integral_time[-1]+odeint_setp, odeint_setp)
    integral_time2 = np.round_(integral_time2, decimals = 2)
    plot_time = list(reversed(integral_time2))
    sample_time = np.where(np.isin(np.array(plot_time),integral_time))[0]
    sample_time = list(reversed(sample_time))
    with torch.no_grad():
        for i in range(len(integral_time)):

            z_t0 =  data_train[i]
            z_t_data.append(z_t0.cpu().detach().numpy())
            t_list2.append(integral_time[i])
        
        all_data = np.vstack([data.cpu().numpy() for data in data_train])
        min_values = np.min(all_data, axis=0)
        max_values = np.max(all_data, axis=0)
        #z_t0 =  Sampling(viz_samples, train_time, len(train_time)-1,data_train,sigma_a,device)
        z_t0 =  ImportanceSampling(viz_samples, train_time, len(train_time)-1,data_train,sigma_a,device)
        logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        g0 = torch.zeros(z_t0.shape).type(torch.float32).to(device)
        for i in range(len(integral_time)):
            viz_data =  ImportanceSampling(viz_samples, train_time, i,data_train,sigma_a,device)
            z_t_data2.append(viz_data.cpu().detach().numpy())
            v_t = func(torch.tensor(integral_time[i]).type(torch.float32).to(device),(viz_data,logp_diff_t0,logp_diff_t0,g0))[0]#-0.04*func(torch.tensor(integral_time[i]).type(torch.float32).to(device),(viz_data,logp_diff_t0,logp_diff_t0,g0))[3]
            v.append(v_t.cpu().detach().numpy())
            
            
        aa=250#3
        widths = 0.2 #arrow width
        ############################################ plot cell velocity
        fig = plt.figure(figsize=(4*2,3*2), dpi=300)
        v_scale = 2
        ax1 = plt.axes()
        ax1.grid(False)
        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.set_xlim(min_values[0]-0.5,max_values[0]+0.5)
        ax1.set_ylim(min_values[1]-0.5,max_values[1]+0.5)
        ax1.set_xticks([min_values[0]-0.5,max_values[0]+0.5])
        ax1.set_yticks([min_values[1]-0.5,max_values[1]+0.5])
        line_width = 0.3
        color_wanted = [np.array([250,187,110])/255,
                        np.array([173,219,136])/255,
                        np.array([250,199,179])/255,
                        np.array([238,68,49])/255,
                        np.array([206,223,239])/255,
                        np.array([3,149,198])/255,
                        np.array([180,180,213])/255,
                        np.array([178,143,237])/255]
          
        for i in range(len(integral_time)):
            ax1.scatter(z_t_data[i][:100,0],z_t_data[i][:100,1],#z_t_data[i][:,2],
                        linewidth=line_width,alpha = 0.7, facecolors=color_wanted[i], s=aa,#'none'
                        edgecolors=color_wanted[i],label=integral_time[i])#,zorder=1)
        
        for i in range(len(integral_time)):
            ax1.scatter(z_t_data2[i][:,0],z_t_data2[i][:,1],#z_t_data[i][:,2],
                        linewidth=line_width,alpha = 0.7, facecolors=color_wanted[i], s=aa,
                        edgecolors=color_wanted[i],label=integral_time[i])#,zorder=1)
      
        # add inferrred trajecotry
        for i in range(len(integral_time)):
            ax1.quiver(z_t_data2[i][:,0],z_t_data2[i][:,1],#z_t_samples[sample_time[i]][:,2],
                       v[i][:,0]/v_scale,v[i][:,1]/v_scale,#v[sample_time[i]][:,2]/v_scale,
                       color='k',alpha=1,linewidths=widths*0.1)#,arrow_length_ratio=0.3,zorder=4)

        plt.savefig(os.path.join(args.save_dir, "est_v_2d_"+args.dataset+"_1.png"),format="png",pad_inches=0.1, bbox_inches='tight',dpi=300)
        plt.show()
                                       
# forward process to predict cell fate decision
def plot_2d_sde_fore(func,data_train,integral_time,args,device,start_point,start_time,end_time,data_type,
                     num=10,ode_setp = 0.5,n=2,classifier='non'):
    
    z_t_samples = []
    z_t_samples2 = []
    odeint_setp = ode_setp
    integral_time2 = np.arange(start_time, end_time+odeint_setp, odeint_setp)
    plot_time = np.round_(integral_time2, decimals = 2)
    D = func.d.detach().cpu().numpy()
    sigma = np.sqrt(2*odeint_setp*D)
    z_t0 =  start_point
    z_t0 = torch.tensor(z_t0).type(torch.float32).to(device) 
    logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
    g0 = torch.zeros(z_t0.shape).type(torch.float32).to(device)
    z_t_samples.append(z_t0.cpu().detach().numpy())
    z_t_samples2.append(z_t0.repeat(num, 1).cpu().detach().numpy())
    dim=z_t0.shape[1]
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})
    options.update({'t0': start_time})
    options.update({'t1': end_time})
    options.update({'t_eval':plot_time})
    z_t1,_, logp_diff_t1,_= odesolve(func,y0=(z_t0,logp_diff_t0,logp_diff_t0,g0),options=options)
    z_t0=torch.tensor(z_t_samples2[0]).to(device) #.cpu()  # Move tensor to CPU
    for i in range(len(plot_time)-1):
        v_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t0,logp_diff_t0,logp_diff_t1,g0))[0] 
        z_t00=z_t0+v_t*odeint_setp+torch.normal(mean=0, std=sigma, size=(num, dim)).to(device)#torch.tensor(np.random.normal(loc=0, scale=sigma, size=num)).to(device)
        z_t0=z_t00
        z_t_samples.append(z_t1[i+1].cpu().detach().numpy())
        z_t_samples2.append(z_t00.cpu().detach().numpy())
    
    z_t_data = [z.cpu().detach().numpy() for z in data_train]
    size=[z.shape[0] for z in z_t_data]
    aa=200#3
    all_data = np.vstack([data.cpu().numpy() for data in data_train])
    min_values = np.min(all_data, axis=0)
    max_values = np.max(all_data, axis=0)
    ############################################ 
    fig = plt.figure(figsize=(4*2,3*2), dpi=200)
    ax1 = plt.axes()
    ax1.grid(False)
    ax1.set_xlabel('PC1', fontsize=20)
    ax1.set_ylabel('PC2', fontsize=20)
    ax1.set_xlim(min_values[0]-0.25,max_values[0]+0.25)
    ax1.set_ylim(min_values[1]-0.25,max_values[1]+0.25)
    ax1.set_xticks([min_values[0]-0.25,max_values[0]+0.25])
    ax1.set_yticks([min_values[1]-0.25,max_values[1]+0.25])
    ax1.tick_params(axis='both', labelsize=15)  
    ax1.axis('off')
    line_width = 0.8
    color_wanted = ['#d62728', '#2ca02c', '#ff7f0e',
                    '#e377c2', '#17becf', '#bcbd22', 
                    '#1f77b4', '#9467bd', '#8c564b', 
                    '#7f7f7f','#2f7f2e']
    
    label_time= ['0d','2d','6d','10d', '2dW', '6dW', '10dW']
    # Randomly sample 'sample_size' indices from the total number of samples
    sample_size=1000
    num_all= min(size)
    sample_indices = torch.randint(0, num_all, (sample_size,))
    unique_data_types = ['E','M','MET','pEMT']
    color_palette = sns.color_palette(color_wanted, len(unique_data_types)) 
    palette = {cell_type: color for cell_type, color in zip(unique_data_types, color_palette)}
    light_blue = np.array([173, 216, 230])/255
    light_red = np.array([244, 104, 104])/255
    dark_blue = np.array([30, 144, 255])/255
    dark_red = np.array([255, 0, 0])/255
    for i in range(len(integral_time)):
        # ax1.scatter(z_t_data[i][sample_indices][:,0],z_t_data[i][sample_indices][:,1],#z_t_data[i][:,2],
        #             s=aa,linewidth=0.8,alpha = 0.7, facecolors='none', #facecolors=color_wanted[i],#
        #             edgecolors=color_wanted[i],label=label_time[i])#,zorder=1)
        current_cell_type = data_type[i][sample_indices]  # 获取当前的cell_type
        current_cell_color = [palette[key] for key in current_cell_type if key in palette]  # 转换为元组以便于哈希
        ax1.scatter(z_t_data[i][sample_indices][:,0],z_t_data[i][sample_indices][:,1],#z_t_data[i][:,2],
                    linewidth=line_width,alpha = 0.7, facecolors=current_cell_color, s=aa,#'none'
                    edgecolors=current_cell_color,label=current_cell_type)#,zorder=1)
        
    ax1.scatter(z_t_samples[0][0][0], z_t_samples[0][0][1], color='black', s=100, zorder=3,label='start')
    for j in range(num): #individual traj
        for i in range(len(plot_time)-1):
            ax1.plot([z_t_samples2[i][j, 0], z_t_samples2[i+1][j, 0]],
                 [z_t_samples2[i][j, 1], z_t_samples2[i+1][j, 1]],
                 linewidth=1.5, color=light_blue, zorder=2)
        ax1.scatter(z_t_samples2[i+1][j, 0], z_t_samples2[i+1][j, 1], color=dark_blue, s=100, zorder=3,label='sde' if j == 0 else "")
    for i in range(len(plot_time)-1):
        ax1.plot([z_t_samples[i][0][0], z_t_samples[i+1][0][0]],
             [z_t_samples[i][0][1], z_t_samples[i+1][0][1]],
             linewidth=2.5, color=light_red, zorder=2) 

    ax1.scatter(z_t_samples[i+1][0][0], z_t_samples[i+1][0][1], color=dark_red, s=100, zorder=3,label='ode')
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cell_type,markerfacecolor=color,  # 设置为透明以实现空心效果'none'
                      markeredgecolor=color, markersize=15,markeredgewidth=2) for cell_type, color in palette.items()]
    ax1.legend(handles=handles, title="Cell Types",fontsize=30,
                frameon=False,title_fontsize=30,loc=(1.02,0.3))         

    #plt.savefig(os.path.join(args.save_dir, "traj_sde_fore_"+args.dataset+"_type_7.pdf"),format="pdf",pad_inches=0.1, bbox_inches='tight')
    plt.savefig(os.path.join(args.save_dir, "traj_sde_fore_"+args.dataset+".png"),format="png",pad_inches=0.1, bbox_inches='tight',dpi=300)
    plt.show()

#backward process to estimate cell ancestry
def plot_2d_sde_back(func,data_train,integral_time,args,device,start_point,start_time,end_time,num=10,ode_setp = 0.5,n=2):
    z_t_samples = []
    z_t_samples2 = []
    odeint_setp = ode_setp
    integral_time2 = np.arange(end_time, start_time+odeint_setp, odeint_setp)
    integral_time2 = np.round_(integral_time2, decimals = 2)
    plot_time = list(reversed(integral_time2))
    D = func.d.detach().cpu().numpy()
    sigma = np.sqrt(2*odeint_setp*D)
    # traj backward
    z_t0 =  start_point
    z_t0 = torch.tensor(z_t0).type(torch.float32).to(device) 
    logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
    g0 = torch.zeros(z_t0.shape).type(torch.float32).to(device)
    z_t_samples.append(z_t0.cpu().detach().numpy())
    z_t_samples2.append(z_t0.repeat(num, 1).cpu().detach().numpy())  # Repeat along the first dimension
    dim=z_t0.shape[1]
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})
    options.update({'t0': start_time})
    options.update({'t1': end_time})
    options.update({'t_eval':plot_time})
    z_t1,_, logp_diff_t1,_= odesolve(func,y0=(z_t0,logp_diff_t0,logp_diff_t0,g0),options=options)
    z_t0=torch.tensor(z_t_samples2[0]).to(device) #.cpu()  # Move tensor to CPU
    for i in range(len(plot_time)-1):
        v_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t0,logp_diff_t0,logp_diff_t1,g0))[0] 
        s_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t0,logp_diff_t0,logp_diff_t1,g0))[3] 
        z_t00=z_t0-1.0*(v_t-func.d*s_t)*odeint_setp-torch.normal(mean=0, std=sigma, size=(num, dim)).to(device)#torch.tensor(np.random.normal(loc=0, scale=sigma, size=num)).to(device)
        z_t0=z_t00
        z_t_samples.append(z_t1[i+1].cpu().detach().numpy())
        z_t_samples2.append(z_t00.cpu().detach().numpy())
    
    z_t_data = [z.cpu().detach().numpy() for z in data_train]
    aa=50#3
    all_data = np.vstack([data.cpu().numpy() for data in data_train])
    min_values = np.min(all_data, axis=0)
    max_values = np.max(all_data, axis=0)
    fig = plt.figure(figsize=(4*2,3*2), dpi=200)
    plt.tight_layout()
    plt.margins(0, 0)
    plt.tight_layout()
    plt.axis('off')
    plt.margins(0, 0)
    ax1 = plt.axes()
    ax1.grid(False)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_xlim(min_values[0]-0.5,max_values[0]+0.5)
    ax1.set_ylim(min_values[1]-0.5,max_values[1]+0.5)
    ax1.set_xticks([min_values[0]-0.5,max_values[0]+0.5])
    ax1.set_yticks([min_values[1]-0.5,max_values[1]+0.5])
    plt.title('The estimated cell velocity',fontsize=30)
    line_width = 0.3
    color_wanted = [np.array([250,187,110])/255,
                    np.array([173,219,136])/255,
                    np.array([250,199,179])/255,
                    np.array([238,68,49])/255,
                    np.array([206,223,239])/255,
                    np.array([3,149,198])/255,
                    np.array([180,180,213])/255,
                    np.array([178,143,237])/255]
    
    light_blue = np.array([173, 216, 230])/255
    light_red = np.array([244, 104, 104])/255
    dark_blue = np.array([30, 144, 255])/255
    dark_red = np.array([255, 0, 0])/255
    ax1.scatter(z_t_samples[0][0][0], z_t_samples[0][0][1], color='black', s=100, zorder=3,label='end')
    for j in range(num): #individual traj
        for i in range(len(plot_time)-1):
            ax1.plot([z_t_samples2[i][j,0],z_t_samples2[i+1][j,0]],
                        [z_t_samples2[i][j,1],z_t_samples2[i+1][j,1]],
                        #[z_t_samples[i][j,2],z_t_samples[i+1][j,2]],
                        linewidth=1.5, color=light_blue,zorder=2)
        ax1.scatter(z_t_samples2[i+1][j, 0], z_t_samples2[i+1][j, 1], color=dark_blue, s=100, zorder=3,label='sde' if j == 0 else "")
    for i in range(len(plot_time)-1):
        ax1.plot([z_t_samples[i][0][0],z_t_samples[i+1][0][0]],
                    [z_t_samples[i][0][1],z_t_samples[i+1][0][1]],
                    linewidth=2.5,color =light_red,zorder=2)
    ax1.scatter(z_t_samples[i+1][0][0], z_t_samples[i+1][0][1], color=dark_red, s=100, zorder=3,label='ode')
    for i in range(len(integral_time)):
        ax1.scatter(z_t_data[i][:,0],z_t_data[i][:,1],#z_t_data[i][:,2],
                    s=aa,linewidth=0.8,alpha = 0.7, facecolors='none', #facecolors=color_wanted[i],#
                    edgecolors=color_wanted[i])#,label=integral_time[i])#,zorder=1)
    plt.legend(fontsize=15)    
    plt.savefig(os.path.join(args.save_dir, "traj_sde_back_"+args.dataset+".pdf"),format="pdf",pad_inches=0.1, bbox_inches='tight',dpi=300)
    plt.show()
    
# identifying trajectory related genes
def plot_de_fore(func,data_train,integral_time,args,device,start_point,start_time,end_time,gene_name,filepath,pca_model,ode_setp = 0.5):
    z_t_samples = []
    odeint_setp = ode_setp
    integral_time2 = np.arange(start_time, end_time+odeint_setp, odeint_setp)
    plot_time = np.round_(integral_time2, decimals = 2)
    z_t0 =  start_point
    z_t0 = torch.tensor(z_t0).type(torch.float32).to(device) 
    logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
    g0 = torch.zeros(z_t0.shape).type(torch.float32).to(device)
    z_t_samples.append(pca_model.inverse_transform(z_t0.cpu().detach().numpy()))#.cpu.detach().numpy()
    dim=z_t0.shape[1]
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})
    options.update({'t0': start_time})
    options.update({'t1': end_time})
    options.update({'t_eval':plot_time})
    z_t1,_, logp_diff_t1,_= odesolve(func,y0=(z_t0,logp_diff_t0,logp_diff_t0,g0),options=options)
    for i in range(len(plot_time)-1):
        z_t_samples.append(pca_model.inverse_transform(z_t1[i+1].cpu().detach().numpy()))#.cpu().detach().numpy()
    
    z_t_samples_np = np.array(z_t_samples).reshape(len(plot_time), len(gene_name))
    plot_time_reshaped = plot_time.reshape(-1, 1)
    combined_data = np.hstack((z_t_samples_np, plot_time_reshaped))  
    correlation_matrix = np.corrcoef(combined_data, rowvar=False)  
    correlation_with_time = correlation_matrix[-1, :-1]  
    sorted_indices = np.argsort(correlation_with_time)[::-1]  
    top_10_indices = sorted_indices[:10]  
    bottom_10_indices = sorted_indices[-10:]  
    pd.DataFrame(data=correlation_with_time[sorted_indices].flatten(), index=[gene_name[i] for i in sorted_indices],
                 columns=['pcc']).to_csv(filepath+'gene\\de_gene_fore.csv')
    top_10_correlations = correlation_with_time[top_10_indices]
    bottom_10_correlations = correlation_with_time[bottom_10_indices]
    # Create dataframes for the heatmaps
    top_10_df = pd.DataFrame(data=top_10_correlations.flatten(), index=[gene_name[i] for i in top_10_indices], columns=['Top 10'])
    bottom_10_df = pd.DataFrame(data=bottom_10_correlations.flatten(), index=[gene_name[i] for i in bottom_10_indices.flatten()], columns=['Bottom 10'])
    # Define the normalization for the color scale
    divnorm1 = colors.TwoSlopeNorm(vmin=-1, vcenter=0., vmax=1)
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(4, 4), dpi=200, sharex=False, sharey=False)
    cbar_ax = fig.add_axes([.91, .3, .02, .4])
    # Plot top 10 heatmap
    sns.heatmap(top_10_df, ax=axes[0], cmap="coolwarm", norm=divnorm1, xticklabels=[], yticklabels=True, annot=False, cbar=False)
    axes[0].set_title('Top 10')
    # Plot bottom 10 heatmap
    sns.heatmap(bottom_10_df, ax=axes[1], cmap="coolwarm", norm=divnorm1, xticklabels=[], yticklabels=True, annot=False, cbar_ax=cbar_ax)
    axes[1].set_title('Bottom 10')
    plt.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig(filepath+'gene\\de_gene_fore.pdf',format="pdf",pad_inches=0.2, bbox_inches='tight')
    plt.show()

## plot gene perturbation result
def plot_gene_fore(func,data_train,integral_time,args,device,start_point,start_time,end_time,
                     gene_name,gene_index,filepath,classifier,pca_model,num=10,ode_setp = 0.5):
    
    odeint_setp = ode_setp
    integral_time2 = np.arange(start_time, end_time+odeint_setp, odeint_setp)
    plot_time = np.round_(integral_time2, decimals = 2)
    D = func.d.detach().cpu().numpy()
    sigma = np.sqrt(2*odeint_setp*D)
    cell_types = ['E','M','MET','pEMT']  
    results = []
    for k in [-3,-2,-1,0,1,2,3]:
        z_t_samples2 = []
        z_t0 =  start_point
        z_t0 = torch.tensor(z_t0).type(torch.float32).to(device) 
        z_t0=pca_model.inverse_transform(z_t0.cpu().detach().numpy())
        z_t0[0,gene_index]=z_t0[0,gene_index]*(10**k)
        z_t0=pca_model.transform(z_t0)
        z_t_samples2.append(z_t0.repeat(num, 0))  # Repeat along the first dimension .cpu().detach().numpy()
        dim=z_t0.shape[1]
        z_t0=torch.tensor(z_t_samples2[0]).type(torch.float32).to(device) #.cpu()  # Move tensor to CPU
        logp_diff_t0 = torch.zeros(z_t0.shape[0], 1).type(torch.float32).to(device)
        g0 = torch.zeros(z_t0.shape).type(torch.float32).to(device)
        for i in range(len(plot_time)-1):
            v_t = func(torch.tensor(plot_time[i+1]).type(torch.float32).to(device),(z_t0,logp_diff_t0,logp_diff_t0,g0))[0] 
            z_t00=z_t0+v_t*odeint_setp+torch.normal(mean=0, std=sigma, size=(num, dim)).to(device)
            z_t0=z_t00
            z_t_samples2.append(z_t00.cpu().detach().numpy())
            
        y_pred = classifier.predict(z_t_samples2[-1])
        counts = {cell_type: np.sum(y_pred == cell_type) for cell_type in cell_types}
        total_predictions = sum(counts.values())
        percentages = {cell_type: (count / total_predictions) * 100 for cell_type, count in counts.items()}
        results.append({'k': k, **percentages})
    
    results_df = pd.DataFrame(results)
    results_df.set_index('k', inplace=True)
    results_melted = results_df.reset_index().melt(id_vars='k', var_name='Cell Type', value_name='Percentage')
    color_wanted = ['#d62728', '#2ca02c', '#ff7f0e',
                    '#e377c2', '#17becf', '#bcbd22', 
                    '#1f77b4', '#9467bd', '#8c564b', 
                    '#7f7f7f','#2f7f2e']
    unique_data_types = ['E','M','MET','pEMT']
    color_palette = sns.color_palette(color_wanted, len(unique_data_types))
    palette = {cell_type: color for cell_type, color in zip(unique_data_types, color_palette)}
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(results_df))
    for cell_type in unique_data_types:
        sns.barplot(data=results_melted[results_melted['Cell Type'] == cell_type], x='k', y='Percentage', bottom=bottom, color=palette[cell_type], label=cell_type)
        bottom += results_melted[results_melted['Cell Type'] == cell_type]['Percentage'].values

    plt.title('Gene Perturbation by ' + gene_name, fontsize=20)
    plt.xlabel(r'Perturbation Level ($\log_{10}$)', fontsize=30)
    plt.ylabel('Percentage (%)', fontsize=30)
    plt.xticks(rotation=0)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(title='Cell Types', title_fontsize=20, loc=(1.02, 0.3), fontsize=20, handletextpad=0.5, markerscale=2)
    plt.tight_layout()
    plt.savefig(filepath + 'gene\\' + gene_name + '_cell_type_prediction_stacked.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    