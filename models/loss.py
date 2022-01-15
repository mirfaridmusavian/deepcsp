import torch
import torch.nn as nn 

class filterbank_csp_loss(nn.Module):

    def __init__(self, n_components:int, n_channels:int, device:str,
                 component_order = "alternate"):
        super(csp_loss,self).__init__()
        self.n_components = n_components
        self.n_channels = n_channels
        self.device = device
        self.component_order = component_order
        
        
    def transform(self, H, filters_, log=True):
        n_features = H.size(2)
        X = H
        X = filters_ @ X
        X = torch.norm(X, dim=2)**2 / n_features

        if log:
            X = torch.log(X)
            
        return X
    
    
    def compute_val_(self, filt, csp, y):
        filt = filt.unsqueeze(1)

        n_channels, n_features = filt.size()[2:]
        X = filt.squeeze(dim=1)

        _classes = torch.unique(y)
        n_classes = len(_classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        covs = torch.zeros((n_classes, n_channels, n_channels)).to(self.device)
        sample_weights = []
        for class_idx, this_class in enumerate(_classes):

            class_ = X[y == this_class]
            for each in range(class_.shape[0]):
                c = torch.mm(class_[each,:,:], class_[each,:,:].t()).to(self.device)
                covs[class_idx] = covs[class_idx] + c / torch.trace(c)

            weight = len(class_)
            sample_weights.append(weight)

        if n_classes == 2:

            
            S_t = covs.sum(0) + torch.eye(n_channels).to(self.device) * 1e-4
            S_w = covs[0] + torch.eye(n_channels).to(self.device) * 1e-4

        loss = torch.trace(csp @ S_w @ csp.T) / torch.trace(csp @ S_t @ csp.T)
        return  1/loss
    

    def compute_loss(self, filt, y):
        filt = filt.unsqueeze(1)

        n_channels, n_features = filt.size()[2:]
        X = filt.squeeze(dim=1)

        _classes = torch.unique(y)
        n_classes = len(_classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        covs = torch.zeros((n_classes, n_channels, n_channels)).to(self.device)
        sample_weights = []
        for class_idx, this_class in enumerate(_classes):

            class_ = X[y == this_class]
            for each in range(class_.shape[0]):
                c = torch.mm(class_[each,:,:], class_[each,:,:].t()).to(self.device)
                covs[class_idx] = covs[class_idx] + c / torch.trace(c)

            weight = len(class_)
            sample_weights.append(weight)

        if n_classes == 2:
            
            S_t = covs.sum(0) + torch.eye(n_channels).to(self.device) * 1e-4
            S_w = covs[0] + torch.eye(n_channels).to(self.device) * 1e-4

            S_t = S_t + torch.eye(S_t.shape[0]).to(self.device)
            L = torch.cholesky(S_t, upper=False)
            P = torch.solve(S_w, L)[0] @ torch.pinverse(L).t()

            eigen_values, eigen_vectors = torch.symeig(P, eigenvectors=True)
            if self.component_order == "mutual_info":
                eigen_values, indices = torch.sort(torch.abs(eigen_values - 0.5), descending=True)
            if self.component_order == "alternate":
                i = torch.argsort(eigen_values)
                i_ = torch.argsort(eigen_values, descending=True)
                indices = torch.empty_like(i)
                if (len(i)%2)==0:
                    indices[1::2] = i[:len(i) // 2]
                    indices[0::2] = i_[:len(i_) // 2]
                else:
                    indices[1::2] = i[:len(i) // 2]
                    indices[0::2] = i_[:(len(i_) // 2)+1]
                
                    
            eigen_vectors = eigen_vectors[:, indices]

            eigen_vectors = torch.solve(eigen_vectors, L.t())[0]

        filters_ = eigen_vectors.t()[:self.n_components]
        
        loss = torch.trace(filters_ @ S_w @ filters_.T) / torch.trace(filters_ @ S_t @ filters_.T)
        return  1/loss, filters_
    
    def filter_bank(self, H, y):
        n_filters = H.shape[1]
        self.filters = torch.zeros((n_filters, self.n_components, self.n_channels)).to(self.device)
        losses = torch.zeros(n_filters).to(self.device)
        transformed = torch.zeros((n_filters, H.shape[0], self.n_components)).to(self.device)
        for ind in range(n_filters):
            losses[ind], self.filters[ind] = self.compute_loss(H[:,ind,:,:], y)
            transformed[ind] = self.transform(H[:,ind,:,:], self.filters[ind])
        return losses.sum(), transformed
    
    def filter_bank_val(self, H, y):
        n_filters = H.shape[1]
        losses = torch.zeros(n_filters).to(self.device)
        transformed = torch.zeros((n_filters, H.shape[0], self.n_components)).to(self.device)
        for ind in range(n_filters):
            losses[ind] = self.compute_val_(H[:,ind,:,:],self.filters[ind], y)
            transformed[ind] = self.transform(H[:,ind,:,:], self.filters[ind])
        return losses.sum(), transformed
    

    def forward(self, H, y):
        return self.filter_bank(H, y)


class csp_loss(nn.Module):

    def __init__(self, n_components:int, n_channels:int, device:str,
                 component_order = "alternate"):
        super(csp_loss,self).__init__()
        self.n_components = n_components
        self.n_channels = n_channels
        self.device = device
        self.component_order = component_order
        
        
    def transform(self, H, filters_, log=True):
        n_features = H.size(2)
        X = H
        X = filters_ @ X
        X = torch.norm(X, dim=2)**2 / n_features

        if log:
            X = torch.log(X)
            
        return X
    
    
    def compute_val_(self, filt, csp, y):
        filt = filt.unsqueeze(1)

        n_channels, n_features = filt.size()[2:]
        X = filt.squeeze(dim=1)

        _classes = torch.unique(y)
        n_classes = len(_classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        covs = torch.zeros((n_classes, n_channels, n_channels)).to(self.device)
        sample_weights = []
        for class_idx, this_class in enumerate(_classes):

            class_ = X[y == this_class]
            for each in range(class_.shape[0]):
                c = torch.mm(class_[each,:,:], class_[each,:,:].t()).to(self.device)
                covs[class_idx] = covs[class_idx] + c / torch.trace(c)

            weight = len(class_)
            sample_weights.append(weight)

        if n_classes == 2:

            
            S_t = covs.sum(0) + torch.eye(n_channels).to(self.device) * 1e-4
            S_w = covs[0] + torch.eye(n_channels).to(self.device) * 1e-4

        loss = torch.trace(csp @ S_w @ csp.T) / torch.trace(csp @ S_t @ csp.T)
        return  1/loss
    

    def compute_loss(self, filt, y):
        filt = filt.unsqueeze(1)

        n_channels, n_features = filt.size()[2:]
        X = filt.squeeze(dim=1)

        _classes = torch.unique(y)
        n_classes = len(_classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        covs = torch.zeros((n_classes, n_channels, n_channels)).to(self.device)
        sample_weights = []
        for class_idx, this_class in enumerate(_classes):

            class_ = X[y == this_class]
            for each in range(class_.shape[0]):
                c = torch.mm(class_[each,:,:], class_[each,:,:].t()).to(self.device)
                covs[class_idx] = covs[class_idx] + c / torch.trace(c)

            weight = len(class_)
            sample_weights.append(weight)

        if n_classes == 2:
            
            S_t = covs.sum(0) + torch.eye(n_channels).to(self.device) * 1e-4
            S_w = covs[0] + torch.eye(n_channels).to(self.device) * 1e-4

            S_t = S_t + torch.eye(S_t.shape[0]).to(self.device)
            L = torch.cholesky(S_t, upper=False)
            P = torch.solve(S_w, L)[0] @ torch.pinverse(L).t()

            eigen_values, eigen_vectors = torch.symeig(P, eigenvectors=True)
            if self.component_order == "mutual_info":
                eigen_values, indices = torch.sort(torch.abs(eigen_values - 0.5), descending=True)
            if self.component_order == "alternate":
                i = torch.argsort(eigen_values)
                i_ = torch.argsort(eigen_values, descending=True)
                indices = torch.empty_like(i)
                if (len(i)%2)==0:
                    indices[1::2] = i[:len(i) // 2]
                    indices[0::2] = i_[:len(i_) // 2]
                else:
                    indices[1::2] = i[:len(i) // 2]
                    indices[0::2] = i_[:(len(i_) // 2)+1]
                
                    
            eigen_vectors = eigen_vectors[:, indices]

            eigen_vectors = torch.solve(eigen_vectors, L.t())[0]

        filters_ = eigen_vectors.t()[:self.n_components]
        
        loss = torch.trace(filters_ @ S_w @ filters_.T) / torch.trace(filters_ @ S_t @ filters_.T)
        return  1/loss, filters_
    

    def valid_step(self, H, y):
        loss = self.compute_val_(H ,self.filters, y)
        transformed = self.transform(H, self.filters)
        
        return loss, transformed

    def forward(self, H, y):
        loss, self.filters = self.compute_loss(H, y)
        transformed = self.transform(H, self.filters)
        return loss, transformed