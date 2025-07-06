# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:29:19 2024

@author: JTliu
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import joblib
sys.path.append('C:\\Users\\JTliu\\Desktop\\DiffusionOT-main\\')
from utility import *
args = create_args()
   
if __name__ == '__main__':
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cpu')#torch.cuda.device_count()
    # load dataset #data_train = loaddata(args,device)
    data = np.load(args.input_dir+'/EMT.npz', allow_pickle=True)
    latent_ae_scaled = data['pca_scaled']
    time_label = data['time_label'].astype(str)
    type_label = data['type_label']
    time_all = ['1','2','3','4']
    data_train = []
    data_type = []
    for k in range(len(time_all)):
        indices = [i for i, l in enumerate(time_label) if l == time_all[k]]
        samples = latent_ae_scaled[indices,]
        cell_type = type_label[indices,]
        samples = torch.from_numpy(samples).type(torch.float32).to(device)
        data_train.append(samples)
        data_type.append(cell_type)
        
    integral_time = args.timepoints
    time_pts = range(len(data_train))
    leave_1_out = []
    train_time = [x for i,x in enumerate(time_pts) if i!=leave_1_out]

    # model
    func = RUOT(in_out_dim=data_train[0].shape[1], hidden_dim=args.hidden_dim,n_hiddens=args.n_hiddens,activation=args.activation,d=args.d).to(device)
    func.apply(initialize_weights)

    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})

    # weight_decay=0.01: 这设置了权重衰减（L2正则化）的系数。权重衰减是一种正则化技术，用于防止模型过拟合，通过在损失函数中添加一个与权重大小成比例的项来实现。
    optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay=0.01)
    lr_adjust = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[
                                               args.niters-400, args.niters-200], gamma=0.5, last_epoch=-1)
 
    mse = nn.MSELoss()

    LOSS = []
    L2_1 = []
    L2_2 = []
    L2_3 = []
    L2_4 = []
    Trans = []
    Sigma = []
    
    
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        ckpt_path = os.path.join(args.save_dir, 'ckpt_{}.pth'.format(args.dataset))
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        ######################################     pretrain score function    ##################################
        # fix func.hyper_net1，hyper_net2
        fixed_layers = [func.hyper_net1, func.hyper_net2]
        func.d.requires_grad = False#True
        for layer in fixed_layers:
            for param in layer.parameters():
                param.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, func.parameters()), lr=args.lr, weight_decay=0.01)

        sigma_now = 1
        loss_score_all=[]
        for itr in range(1, 200 + 1):#args.niters
            optimizer.zero_grad()     
            loss_score= pre_train_score(mse,func,args,data_train,train_time,integral_time,sigma_now,device,itr)
            loss_score.backward()
            optimizer.step()
            print('Pre_train_score_Iter: {}, loss: {:.4f}'.format(itr, loss_score.item()))
            loss_score_all.append(loss_score.item())
                    
        ##save pre_train result        
        ckpt_path = os.path.join(args.save_dir, 'ckpt_{}_score.pth'.format(args.dataset))
        torch.save({'func_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
        print('Iter {}, Stored ckpt at {}'.format(itr, ckpt_path))       
        
        ######################################     pretrain velocity and growth   ##################################
        fixed_layers = [func.hyper_net3]
        train_layers = [func.hyper_net1, func.hyper_net2]
        func.d.requires_grad = False#True
        for layer in fixed_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in train_layers:
            for param in layer.parameters():
                param.requires_grad = True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, func.parameters()), lr=args.lr, weight_decay=0.01)#args.lr
        sigma_now = 1
        l=1
        for itr in range(1,  100+ 1):#args.niters
            optimizer.zero_grad()
            loss, sigma_now, L2_value3,L2_value4= pre_train_model(mse,func,args,data_train,train_time,integral_time,sigma_now,options,device,itr)
            loss.backward()
            optimizer.step()
            lr_adjust.step()
            l=loss.item()
            LOSS.append(loss.item())
            Sigma.append(sigma_now)
            L2_3.append(L2_value3.tolist())     
            print('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))
                
        ckpt_path = os.path.join(args.save_dir, 'ckpt_{}_pre.pth'.format(args.dataset,itr))
        torch.save({'func_state_dict': func.state_dict()}, ckpt_path)
        print('Iter {}, Stored ckpt at {}'.format(itr, ckpt_path))
        
        ##################################  train total loss
        train_layers = [func.hyper_net1, func.hyper_net2,func.hyper_net3]
        func.d.requires_grad = False#True
        for layer in train_layers:
            for param in layer.parameters():
                param.requires_grad = True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, func.parameters()), lr=args.lr, weight_decay=0.01)#args.lr

        sigma_now = 0.5
        D_all=[]
        for itr in range(1, args.niters + 1):#args.niters
            optimizer.zero_grad()
            loss, loss1, sigma_now, L2_value1, L2_value2, L2_value3, L2_value4= train_model(mse,func,args,data_train,train_time,integral_time,sigma_now,options,device,itr)
            loss.backward()
            optimizer.step()
            lr_adjust.step()      
            LOSS.append(loss.item())
            Trans.append(loss1[-1].mean(0).item())
            Sigma.append(sigma_now)
            L2_1.append(L2_value1.tolist())
            L2_2.append(L2_value2.tolist())
            L2_3.append(L2_value3.tolist())
            L2_4.append(L2_value4.tolist())
            print('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))
            if itr % 500 == 0:
                D_t=diffusion_fit(func,args,data_train,train_time,integral_time,device,time_tt=0.01)
                D=torch.mean(D_t)
                D_all.append(D.detach().cpu().numpy())
                ckpt_path = os.path.join(args.save_dir, 'ckpt_{}_itr{}.pth'.format(args.dataset,itr))
                torch.save({'func_state_dict': func.state_dict()}, ckpt_path)
                print('Iter {}, Stored ckpt at {}'.format(itr, ckpt_path))
                    
            if (itr%100 == 0) and (itr<500):
                D_t=diffusion_fit(func,args,data_train,train_time,integral_time,device,time_tt=0.01)
                D=torch.mean(D_t)
                D_all.append(D.detach().cpu().numpy())
                func.d=torch.nn.Parameter(D)  
            

    except KeyboardInterrupt:
        if args.save_dir is not None:
            ckpt_path = os.path.join(args.save_dir, 'ckpt_{}.pth'.format(args.dataset))
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))
    
    
    ckpt_path = os.path.join(args.save_dir, 'ckpt_{}.pth'.format(args.dataset))
    torch.save({
        'func_state_dict': func.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'LOSS':LOSS,
        'TRANS':Trans,
        'L2_1': L2_1,
        'L2_2': L2_2,
        'L2_3': L2_3,
        'L2_4': L2_4,
        'Sigma': Sigma,
        'D':D_all,
    }, ckpt_path)
    print('Stored ckpt at {}'.format(ckpt_path))