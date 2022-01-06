from sklearn.model_selection import train_test_split

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import numpy as np
from moabb.paradigms import MotorImagery
from moabb import datasets
from sklearn import preprocessing
from braindecode.datasets import MOABBDataset


import torch
import torch.nn as nn 
from torch import tensor

import wandb

from utils import utils
from utils import data_utils
from utils.model_utils import get_config, get_optim
from models.model import *
from models.loss import csp_loss

import optuna
import argparse

utils.set_seed(0)
device = utils.set_device()


my_parser = argparse.ArgumentParser()

my_parser.add_argument('--sub', action='store', type=int, default=3)
my_parser.add_argument('--num_epochs', action='store', type=int, default=100)
my_parser.add_argument('--n_trials', action='store', type=int, default=50)
my_parser.add_argument('--dataset', action='store', type=str, default="BNCI2014004")


args = my_parser.parse_args()

# data config
dataset_name=args.dataset
subject_id=args.sub

paradigm = MotorImagery(n_classes=2)
dataset = getattr(datasets, dataset_name)
subjects = [args.sub]

# load data
X, y, metadata = paradigm.get_data(dataset=dataset(), subjects=subjects)
X = X * 1000
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# load metadata
ds = MOABBDataset(dataset_name=dataset_name, subject_ids=subject_id)
ch_names = ds.datasets[0].raw.info["ch_names"]
samplerate = ds.datasets[0].raw.info["sfreq"]

num_epochs = args.num_epochs

trial_num = -1 # trial number of optuna
num_channels = X.shape[1] # for deepcsp 
def objective(trial):
    global trial_num
    trial_num += 1
    #hyperparameters
    n_components = 2
    component_order = 'alternate'
    
    # get parameters specific to the current trial
    
    config = get_config(trial=trial)
        
    # definde model and classifier
    clf = Classifier(n_components, config['filters']).to(device)
    model = Tception(samplerate, config['filters']).to(device)
    CSPloss = csp_loss(n_components = n_components,
                       n_channels = num_channels,
                       device=device,
                       component_order = component_order)

    # get optimizers specific to the current trial
    
    optimizer_model, optimizer_clf = get_optim(config, model, clf)
    criterion = nn.BCELoss()    

    params = {'n_components': n_components,
          'filters': config['filters'],
          'component_order': component_order,
          'optim_type_model': config['optimizer_name_model'],
          'weight_decay_model': config['weight_decay_model'],
          'lr_model': config['lr_model'],
          'optim_type_clf': config['optimizer_name_clf'],
          'weight_decay_clf': config['weight_decay_clf'],
          'lr_clf': config['lr_clf'],
          'trial_id': trial_num}


    val_losses, accs = [], []   
    splitter = getattr(data_utils, "partition_"+dataset_name)
    X_, X_test_, y_, y_test_, folds = splitter(X, y, metadata)
    fold_counter = 0
    for run_ind, (train_index, val_ind) in enumerate(folds):
        fold_counter += 1
        
        # Simulate launching multiple different jobs that log to the same experiment


        # Set group and job_type to see auto-grouping in the UI
        wandb.init(name='trial_'+str(trial_num+1)+"_fold_" + str(fold_counter),
            project=dataset_name+f"_sub_{subject_id}",
            group="trial_" + str(trial_num+1), 
            job_type="fold_" + str(fold_counter))
        wandb.config.update(params)

        X_train, X_valid = X_[train_index], X_[val_ind]
        y_train, y_valid = y_[train_index], y_[val_ind]

        X_train = torch.as_tensor(X_train).float().unsqueeze(1).to(device)
        y_train = torch.as_tensor(y_train).float().to(device)
        X_val = torch.as_tensor(X_valid).float().unsqueeze(1).to(device)
        y_val = torch.as_tensor(y_valid).float().to(device)
        X_test = torch.as_tensor(X_test_).float().unsqueeze(1).to(device)
        y_test = torch.as_tensor(y_test_).float().to(device)
        
        clf.reset_parameters()
        model.reset_parameters()

        for epoch in tqdm(range(num_epochs)):

            
            model.train()
            clf.train()

            optimizer_model.zero_grad()

            H = model(X_train)
            cpsloss_train , transformed = CSPloss(H, y_train)

            cpsloss_train.backward(retain_graph=True)
            optimizer_clf.zero_grad()

            outputs = clf(transformed.clone().detach())
            loss_train = criterion(outputs, y_train)


            loss_train.backward()
            optimizer_model.step()
            optimizer_clf.step()

            accuracy_train = ((outputs > 0.5) == y_train).float().mean()
            model.eval()
            clf.eval()

            with torch.no_grad():
                H = model(X_val)

                cpsloss_val, transformed = CSPloss.filter_bank_val(H, y_val)

                outputs = clf(transformed.clone().detach())
                loss_val = criterion(outputs, y_val)
                accuracy_val = ((outputs > 0.5) == y_val).float().mean()

                H = model(X_test)

                cpsloss_test, transformed = CSPloss.filter_bank_val(H, y_test)

                outputs = clf(transformed.clone().detach())
                loss_test = criterion(outputs, y_test)
                accuracy_test = ((outputs > 0.5) == y_test).float().mean()


            val_losses.append(loss_val)
            accs.append(accuracy_test)
            wandb.log({
                'train_accuracy': accuracy_train,
                'train_loss': loss_train,
                'train_csploss': cpsloss_train,
                'valid_accuracy': accuracy_val,
                'valid_loss': loss_val,
                'valid_csploss': cpsloss_val,
                'test_accuracy': accuracy_test,
                'test_loss': loss_test,
                'test_csploss': cpsloss_test,
                'run': run_ind,
                'epoch': epoch
            })
        

    loss, acc = tensor(val_losses), tensor(accs)
    loss, acc = loss.view(fold_counter, num_epochs), acc.view(fold_counter, num_epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(fold_counter, dtype=torch.long), argmin]
    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    wandb.log({
        'cv_acc_mean': acc_mean,
        'cv_acc_std': acc_std,
        'cv_loss_mean': loss_mean,
        'trial': trial_num
        })

    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()


    return loss_mean

wandb.login()

study = optuna.create_study(direction="minimize", study_name="my_study")
study.optimize(objective, n_trials=args.n_trials)