import torch
import numpy as np

def set_seed(seed_num:int):
    ## determisitic setting for reproducibility
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_num)

def set_device():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device