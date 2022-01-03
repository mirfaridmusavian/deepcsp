import torch
import torch.nn as nn 


class Classifier(nn.Module):
    def __init__(self, n_components, n_filters):
        super(Classifier, self).__init__()
        
        self.cl = nn.Sequential(
            nn.Linear(n_filters*n_components, n_filters),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_filters, 1),
            nn.Sigmoid())
    def reset_parameters(self):
        for each in self.cl:
            if hasattr(each, 'reset_parameters'):
                each.reset_parameters()

        
        
    def forward(self, X):
        x = torch.transpose(X, 0,1)
        out = self.cl(x.reshape(x.shape[0], -1))
        out = out.squeeze()
        return out
    

class Tception(nn.Module):
    def __init__(self, sampling_rate, num_T):
        # input_size: channel x datapoint
        super(Tception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625, 0.03125]
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[0]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[1]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1,int(self.inception_window[2]*sampling_rate)), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,16), stride=(1,16)))
        
        self.BN_t = nn.BatchNorm2d(num_T)
        
    def reset_parameters(self):
        for each in self.Tception1:
            if hasattr(each, 'reset_parameters'):
                each.reset_parameters()
        for each in self.Tception2:
            if hasattr(each, 'reset_parameters'):
                each.reset_parameters()                
        for each in self.Tception3:
            if hasattr(each, 'reset_parameters'):
                each.reset_parameters()
        self.BN_t.reset_parameters()

        
    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out,y),dim = -1)
        y = self.Tception3(x)
        out = torch.cat((out,y),dim = -1)
        out = self.BN_t(out)
        return out