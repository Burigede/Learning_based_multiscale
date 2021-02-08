import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import scipy.io
import h5py

import matplotlib.pyplot as plt
import pickle


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def forward(self, x, y):
        return self.rel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)


class PCA(object):
    def __init__(self, x, dim, subtract_mean=True):
        super(PCA, self).__init__()

        # Input size
        x_size = list(x.size())

        # Input data is a matrix
        assert len(x_size) == 2

        # Reducing dimension is less than the minimum of the
        # number of observations and the feature dimension
        assert dim <= min(x_size)

        self.reduced_dim = dim

        if subtract_mean:
            self.x_mean = torch.mean(x, dim=0).view(1, -1)
        else:
            self.x_mean = torch.zeros((x_size[1],), dtype=x.dtype, layout=x.layout, device=x.device)

        # SVD
        U, S, V = torch.svd(x - self.x_mean)
        V = V.t()

        # Flip sign to ensure deterministic output
        max_abs_cols = torch.argmax(torch.abs(U), dim=0)
        signs = torch.sign(U[max_abs_cols, range(U.size()[1])]).view(-1, 1)
        V *= signs

        self.W = V.t()[:, 0:self.reduced_dim]
        self.sing_vals = S.view(-1, )

    def cuda(self):
        self.W = self.W.cuda()
        self.x_mean = self.x_mean.cuda()
        self.sing_vals = self.sing_vals.cuda()

    def encode(self, x):
        return (x - self.x_mean).mm(self.W)

    def decode(self, x):
        return x.mm(self.W.t()) + self.x_mean

    def forward(self, x):
        return self.decode(self.encode(x))

    def __call__(self, x):
        return self.forward(x)


class LeastSquares(object):
    def __init__(self, x, y, bias=False, lam=0.0, cuda=True):
        super(LeastSquares, self).__init__()

        self.bias = bias
        self.cuda = cuda

        # Input sizes
        x_size = list(x.size())
        y_size = list(y.size())

        # Input data are matricies
        assert len(x_size) == 2 and len(y_size) == 2

        # Numer of observations match
        assert x_size[0] == y_size[0]

        x = x.cpu().numpy()
        y = y.cpu().numpy()

        if bias:
            x = np.append(x, np.ones((x_size[0], 1)), axis=1)

        if lam <= 0.0:
            self.W = torch.from_numpy(np.linalg.lstsq(x, y, rcond=None)[0])
        else:
            myid = lam * np.identity(x.shape[1])
            if bias:
                myid[x.shape[1] - 1, x.shape[1] - 1] = 0

            self.W = torch.from_numpy(np.linalg.solve(x.T.dot(x) + myid, x.T.dot(y)))

        self.W = self.W.type(torch.FloatTensor)
        if cuda:
            self.W = self.W.cuda()

    def forward(self, x):
        if self.bias:
            ones = torch.ones((x.size()[0], 1)).type(torch.FloatTensor)
            if self.cuda:
                ones = ones.cuda()

            return torch.cat((x, ones), dim=1).mm(self.W)

        return x.mm(self.W)

    def __call__(self, x):
        return self.forward(x)


class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


class UnitGaussianNormalizer(object):
    def __init__(self, x, zeroone_first=True):
        super(UnitGaussianNormalizer, self).__init__()

        self.zeroone_first = zeroone_first

        if self.zeroone_first:
            self.min = torch.min(x, 0)[0].view(-1, )
            self.max = torch.max(x, 0)[0].view(-1, )

            s = x.size()
            x = ((x.view(s[0], -1) - self.min) / (self.max - self.min)).view(s)

        self.mean = torch.mean(x, 0).view(-1, )
        self.std = torch.std(x, 0).view(-1, )

    def encode(self, x):
        s = x.size()

        x = x.view(s[0], -1)

        if self.zeroone_first:
            x = (x - self.min) / (self.max - self.min)

        x = (x - self.mean) / self.std

        x = x.view(s)

        return x

    def decode(self, x):
        s = x.size()

        x = x.view(s[0], -1)

        x = (x * self.std) + self.mean

        if self.zeroone_first:
            x = (x * (self.max - self.min)) + self.min

        x = x.view(s)

        return x


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

class PCA_3d(object):
    def __init__(self, x, dim):
        super(PCA_3d, self).__init__()

        # Input size
        x_size = list(x.size())

        # Reducing dimension is less than the minimum of the
        # number of observations and the feature dimension
        #assert dim <= min(x_size)

        self.reduced_dim = dim

        # SVD for 3 inputs
        U1, S1, V1 = torch.svd(x[:, :, 0])
        U2, S2, V2 = torch.svd(x[:, :, 1])
        U3, S3, V3 = torch.svd(x[:, :, 2])
        U4, S4, V4 = torch.svd(x[:, :, 3])
        U5, S5, V5 = torch.svd(x[:, :, 4])
        U6, S6, V6 = torch.svd(x[:, :, 5])
        V1 = V1.t()
        V2 = V2.t()
        V3 = V3.t()
        V4 = V4.t()
        V5 = V5.t()
        V6 = V6.t()

        # Flip sign to ensure deterministic output
        max_abs_cols_1 = torch.argmax(torch.abs(U1), dim=0)
        max_abs_cols_2 = torch.argmax(torch.abs(U2), dim=0)
        max_abs_cols_3 = torch.argmax(torch.abs(U3), dim=0)
        max_abs_cols_4 = torch.argmax(torch.abs(U4), dim=0)
        max_abs_cols_5 = torch.argmax(torch.abs(U5), dim=0)
        max_abs_cols_6 = torch.argmax(torch.abs(U6), dim=0)

        signs_1 = torch.sign(U1[max_abs_cols_1, range(U1.size()[1])]).view(-1, 1)
        signs_2 = torch.sign(U2[max_abs_cols_2, range(U2.size()[1])]).view(-1, 1)
        signs_3 = torch.sign(U3[max_abs_cols_3, range(U3.size()[1])]).view(-1, 1)
        signs_4 = torch.sign(U4[max_abs_cols_4, range(U4.size()[1])]).view(-1, 1)
        signs_5 = torch.sign(U5[max_abs_cols_5, range(U5.size()[1])]).view(-1, 1)
        signs_6 = torch.sign(U6[max_abs_cols_6, range(U6.size()[1])]).view(-1, 1)

        V1 *= signs_1
        V2 *= signs_2
        V3 *= signs_3
        V4 *= signs_4
        V5 *= signs_5
        V6 *= signs_6

        self.W1 = V1.t()[:, 0:self.reduced_dim]
        self.W2 = V2.t()[:, 0:self.reduced_dim]
        self.W3 = V3.t()[:, 0:self.reduced_dim]

        self.W4 = V4.t()[:, 0:self.reduced_dim]
        self.W5 = V5.t()[:, 0:self.reduced_dim]
        self.W6 = V6.t()[:, 0:self.reduced_dim]


    def cuda(self):
        self.W1 = self.W1.cuda()
        self.W2 = self.W2.cuda()
        self.W3 = self.W3.cuda()
        self.W4 = self.W4.cuda()
        self.W5 = self.W5.cuda()
        self.W6 = self.W6.cuda()

    def encode(self, x):
        x1 = x[:,:,0].mm(self.W1)
        x2 = x[:,:,1].mm(self.W2)
        x3 = x[:,:,2].mm(self.W3)
        x4 = x[:,:,3].mm(self.W4)
        x5 = x[:,:,4].mm(self.W5)
        x6 = x[:,:,5].mm(self.W6)
        x  = torch.cat((x1.unsqueeze(2),x2.unsqueeze(2),x3.unsqueeze(2),x4.unsqueeze(2),x5.unsqueeze(2),x6.unsqueeze(2)),2)
        return x

    def decode(self, x):
        x1 = x[:,:,0].mm(self.W1.t())
        x2 = x[:,:,1].mm(self.W2.t())
        x3 = x[:,:,2].mm(self.W3.t())
        x4 = x[:,:,3].mm(self.W4.t())
        x5 = x[:,:,4].mm(self.W5.t())
        x6 = x[:,:,5].mm(self.W6.t())
        x  = torch.cat((x1.unsqueeze(2),x2.unsqueeze(2),x3.unsqueeze(2),x4.unsqueeze(2),x5.unsqueeze(2),x6.unsqueeze(2)),2)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))

    def __call__(self, x):
        return self.forward(x)

class UnitGaussianNormalizer_3d(object):
    def __init__(self, x, zeroone_first=True):
        super(UnitGaussianNormalizer_3d, self).__init__()
        x1 = x[:,:,0]
        x2 = x[:,:,1]
        x3 = x[:,:,2]
        self.zeroone_first = zeroone_first

        if self.zeroone_first:
            self.min_x1 = torch.min(x1, 0)[0].view(-1, )
            self.min_x2 = torch.min(x1, 0)[0].view(-1, )
            self.min_x3 = torch.min(x3, 0)[0].view(-1, )

            self.max_x1 = torch.max(x2, 0)[0].view(-1, )
            self.max_x2 = torch.max(x2, 0)[0].view(-1, )
            self.max_x3 = torch.max(x2, 0)[0].view(-1, )

            s = x1.size()
            x1 = ((x1.view(s[0], -1) - self.min_x1) / (self.max_x1 - self.min_x1)).view(s)
            x2 = ((x2.view(s[0], -1) - self.min_x2) / (self.max_x2 - self.min_x2)).view(s)
            x3 = ((x3.view(s[0], -1) - self.min_x3) / (self.max_x3 - self.min_x3)).view(s)

        self.mean_x1 = torch.mean(x1, 0).view(-1, )
        self.mean_x2 = torch.mean(x2, 0).view(-1, )
        self.mean_x3 = torch.mean(x3, 0).view(-1, )

        self.std_x1 = torch.std(x1, 0).view(-1, )
        self.std_x2 = torch.std(x2, 0).view(-1, )
        self.std_x3 = torch.std(x3, 0).view(-1, )

    def encode(self, x):
        x1 = x[:,:,0]
        x2 = x[:,:,1]
        x3 = x[:,:,2]
        s = x1.size()

        x1 = x1.view(s[0], -1)
        x2 = x2.view(s[0], -1)
        x3 = x3.view(s[0], -1)

        if self.zeroone_first:
            x1 = (x1 - self.min_x1) / (self.max_x1 - self.min_x1)
            x2 = (x2 - self.min_x2) / (self.max_x2 - self.min_x2)
            x3 = (x3 - self.min_x3) / (self.max_x3 - self.min_x3)

        x1 = (x1 - self.mean_x1) / self.std_x1
        x2 = (x2 - self.mean_x2) / self.std_x2
        x3 = (x3 - self.mean_x3) / self.std_x3


        x1 = x1.view(s)
        x2 = x2.view(s)
        x3 = x3.view(s)
        x  = torch.cat((x1.unsqueeze(2),x2.unsqueeze(2),x3.unsqueeze(2)),2)

        return x

    def decode(self, x):
        x1 = x[:,:,0]
        x2 = x[:,:,1]
        x3 = x[:,:,2]
        s = x1.size()

        x1 = x1.view(s[0], -1)
        x2 = x2.view(s[0], -1)
        x3 = x3.view(s[0], -1)

        x1 = (x1 * self.std_x1) + self.mean_x1
        x2 = (x2 * self.std_x2) + self.mean_x2
        x3 = (x3 * self.std_x3) + self.mean_x3

        if self.zeroone_first:
            x1 = (x1 * (self.max_x1 - self.min_x1)) + self.min_x1
            x2 = (x2 * (self.max_x2 - self.min_x2)) + self.min_x2
            x3 = (x3 * (self.max_x3 - self.min_x3)) + self.min_x3

        x1 = x1.view(s)
        x2 = x2.view(s)
        x3 = x3.view(s)
        x  = torch.cat((x1.unsqueeze(2),x2.unsqueeze(2),x3.unsqueeze(2)),2)
        return x
class DenseNet_tensor(nn.Module):
    def __init__(self, ten_layers, layers, nonlinearity):
        super(DenseNet_tensor, self).__init__()
        # =========== change 3d tensor to 1d scalar ================#
        self.n_ten = len(ten_layers)-1
        self.ten_layers = nn.ModuleList()
        for k in range(self.n_ten):
            self.ten_layers.append(nn.Linear(ten_layers[k], ten_layers[k+1]))
            self.ten_layers.append(nonlinearity())

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = nn.ModuleList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, m in enumerate(self.ten_layers):
            x = m(x)
        x = x.squeeze(2)
        for _, l in enumerate(self.layers):
            x = l(x)

        return x
########### TO CHANGE ######################################
USE_CUDA = True
TRAIN_PATH = '~\Taylor3D.mat'

Ntotal     = 35747
train_size = 4000
test_start = 30000

N_test = Ntotal-test_start

F11_FIELD = 'F11_field'
F22_FIELD = 'F22_field'
F12_FIELD = 'F12_field'
F13_FIELD = 'F13_field'
F23_FIELD = 'F23_field'
F33_FIELD = 'F33_field'

SIG_FIELD = 'sig11_field'

SIG11_FIELD = 'sig11_field'
SIG22_FIELD = 'sig22_field'
SIG12_FIELD = 'sig12_field'
SIG13_FIELD = 'sig13_field'
SIG23_FIELD = 'sig23_field'
SIG33_FIELD = 'sig33_field'



grid_size =64   # Only for plotting

loss_func = LpLoss()
######### Preprocessing data ####################
temp = torch.zeros(Ntotal,1)

data_loader = MatReader(TRAIN_PATH)
data_F11  = data_loader.read_field(F11_FIELD).contiguous().view(Ntotal, -1)
data_F22  = data_loader.read_field(F22_FIELD).contiguous().view(Ntotal, -1)
data_F12  = data_loader.read_field(F12_FIELD).contiguous().view(Ntotal, -1)
data_F13  = data_loader.read_field(F13_FIELD).contiguous().view(Ntotal, -1)
data_F23  = data_loader.read_field(F23_FIELD).contiguous().view(Ntotal, -1)
data_F33  = data_loader.read_field(F33_FIELD).contiguous().view(Ntotal, -1)

data_F11  = torch.cat((temp+1,data_F11+1),1)
data_F22  = torch.cat((temp+1,data_F22+1),1)
data_F33  = torch.cat((temp+1,data_F33+1),1)
data_F12  = torch.cat((temp,data_F12),1)
data_F13  = torch.cat((temp,data_F13),1)
data_F23  = torch.cat((temp,data_F23),1)

#data_F11 = data_F11 - 1.0
#data_F22 = data_F22 - 1.0
#data_F33 = data_F33 - 1.0

#data_F11  = data_F11[:,0::10]
#data_F22  = data_F22[:,0::10]
#data_F12  = data_F12[:,0::10]
#data_F13  = data_F13[:,0::10]
#data_F23  = data_F23[:,0::10]
#data_F33  = data_F33[:,0::10]

data_S11  = data_loader.read_field(SIG11_FIELD).contiguous().view(Ntotal, -1)
data_S22  = data_loader.read_field(SIG22_FIELD).contiguous().view(Ntotal, -1)
data_S12  = data_loader.read_field(SIG12_FIELD).contiguous().view(Ntotal, -1)
data_S13  = data_loader.read_field(SIG13_FIELD).contiguous().view(Ntotal, -1)
data_S23  = data_loader.read_field(SIG23_FIELD).contiguous().view(Ntotal, -1)
data_S33  = data_loader.read_field(SIG33_FIELD).contiguous().view(Ntotal, -1)

data_S11  = torch.cat((temp,data_S11),1)
data_S22  = torch.cat((temp,data_S22),1)
data_S13  = torch.cat((temp,data_S13),1)
data_S12  = torch.cat((temp,data_S12),1)
data_S23  = torch.cat((temp,data_S23),1)
data_S33  = torch.cat((temp,data_S33),1)

data_input = torch.cat((data_F11.unsqueeze(2),data_F22.unsqueeze(2),data_F33.unsqueeze(2),data_F12.unsqueeze(2),data_F23.unsqueeze(2),data_F13.unsqueeze(2)),2)

data_output = torch.cat((data_S11.unsqueeze(2),data_S22.unsqueeze(2),data_S33.unsqueeze(2),data_S12.unsqueeze(2),data_S23.unsqueeze(2),data_S13.unsqueeze(2)),2)

#============================ PCA on output and input data ====================================#
x_train = data_input[0:train_size,:,:]
y_train = data_output[0:train_size,:]

x_test = data_input[test_start:Ntotal,:,:]
y_test  = data_output[test_start:Ntotal,:]
# Reduced dimension of inputs
d1 = 100
# Reduced dimension of outputs
d2 = 100

NeuroArchi  = [d1*6, 1000, 1000, 1000,1000, d2*6]
#TensorArchi = [3,256,256,256,1]
# Perform PCA on the CPU (can be a memory bottleneck for the GPU)

x_pca = PCA_3d(x_train, d1)

y_pca = PCA_3d(y_train, d2)

#y_pca = PCA(y_train, d2, subtract_mean=False)

# Move data and models to the GPU
if USE_CUDA:
    x_train = x_train.cuda()
    x_test = x_test.cuda()
    y_train = y_train.cuda()
    y_test  = y_test.cuda()

    x_pca.cuda()
    y_pca.cuda()


x_train_enc = x_pca.encode(x_train)
x_test_enc  = x_pca.encode(x_test)

y_train_enc = y_pca.encode(y_train)
y_test_enc = y_pca.encode(y_test)


net = torch.load('Taylor3Dmax05_nonormal')

if USE_CUDA:
    net.cuda()

# Number of training epochs
epochs = 3000

# Optimizer and learning rate scheduler
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)

# Batch size
b_size = 16

# Wrap traning data in loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_enc, y_train_enc), batch_size=b_size,
                                           shuffle=True)
x_test_enc = x_test_enc.view(-1, d1*6)
# Train neural net

y_test_approx = y_pca.decode(net(x_test_enc).view(-1,d2,6).detach())
print('Relative approximation error (NN):\t', loss_func(y_test_approx, y_test).item())
