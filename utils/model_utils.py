import torch

def get_config(trial):
    config = {}
    config['filters'] = trial.suggest_categorical('filters', [10,15,20])
    optimizer_names = ['adam', 'momentum']
    config['optimizer_name_model'] = trial.suggest_categorical('optim_type_model', optimizer_names)
    config['weight_decay_model'] = trial.suggest_loguniform('weight_decay_model', 1e-8, 1e-3)
    config['lr_model'] = trial.suggest_loguniform('lr_model', 1e-5, 1e-1)


    config['optimizer_name_clf'] = trial.suggest_categorical('optim_type_clf', optimizer_names)
    config['weight_decay_clf'] = trial.suggest_loguniform('weight_decay_clf', 1e-8, 1e-3)
    config['lr_clf'] = trial.suggest_loguniform('lr_clf', 1e-5, 1e-1)

    return config

def get_optim(config, model, clf):
    optimizer_names = ['adam', 'momentum']
    if config['optimizer_name_model'] == optimizer_names[0]:
        optimizer_model = torch.optim.Adam(model.parameters(),
                                     lr=config['lr_model'],
                                     weight_decay=config['weight_decay_model'],
                                     amsgrad=True)
    elif config['optimizer_name_model'] == optimizer_names[1]:
        optimizer_model = torch.optim.SGD(model.parameters(),
                                    lr=config['lr_model'],
                                    momentum=0.9,
                                    weight_decay=config['weight_decay_model'])

        
    if config['optimizer_name_clf'] == optimizer_names[0]:
        optimizer_clf = torch.optim.Adam(clf.parameters(),
                                     lr=config['lr_clf'],
                                     weight_decay=config['weight_decay_clf'],
                                     amsgrad=True)
    elif config['optimizer_name_clf'] == optimizer_names[1]:
        optimizer_clf = torch.optim.SGD(clf.parameters(),
                                    lr=config['lr_clf'],
                                    momentum=0.9,
                                    weight_decay=config['weight_decay_clf'])

    return optimizer_model, optimizer_clf
