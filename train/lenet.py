
# code is from https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py

'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .init_utils import weights_init 

class LeNet(nn.Module):
    def __init__(self, input_dim=32):
        super(LeNet, self).__init__()
        layers = []
        layers += [nn.Conv2d(3, 6, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        layers += [nn.Conv2d(6, 16, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.conv_features = nn.Sequential(*layers)
        layers = []
        layers += [nn.Linear(16*5*5, 120), nn.ReLU(inplace=True)]
        layers += [nn.Linear(120, 84), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        
        self.classifier  = nn.Sequential(nn.Linear(84,10))
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1   = nn.Linear(16*5*5, 120)
        #self.fc2   = nn.Linear(120, 84)
        #self.fc3   = nn.Linear(84, 10)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
        #import pdb;pdb.set_trace()
        #self.apply(weights_init)

    def forward(self, x):
        #out = F.relu(self.conv1(x))
        #out = F.max_pool2d(out, 2)
        #out = F.relu(self.conv2(out))
        #out = F.max_pool2d(out, 2)
        out = self.conv_features(x)
        out = out.view(out.size(0), -1)
        out = self.features(out)
        out = self.classifier(out)
        #out = F.relu(self.fc1(out))
        #out = F.relu(self.fc2(out))
        #out = self.fc3(out)
        #return out

        a = out
        return F.log_softmax(out, dim=1), a

def sanity_test_conv():
    cout,hout,wout = 2,2,2
    cin, hin, win  = 3,4,4
    ck = 3
    stride = 1
    
    conv = nn.Conv2d(cin,cout,ck)
    #conv.weight.data.fill_(1.0)
    #conv.bias.data.fill_(0.0)
    conv.weight.data = torch.arange(cout*cin*ck*ck).reshape(cout, cin, ck, ck).float()
    #conv.bias.data.fill_(0.0)

    w = conv.weight.data
    b = conv.bias.data
    
    data = torch.arange(cin*hin*win).reshape(cin,hin,win).float()
    out  = conv(data[None,:,:,:]).squeeze(dim=0) # 2x2x2
    import pdb;pdb.set_trace()
    

    out2 = torch.zeros([cout, hout, wout])
    for i in range(cout):
        for j in range(hout):
            h_in_ = j
            for k in range(wout):
                w_in_ = k
                out2[i,j,k]=b[i]
                for k1 in range(cin):
                    for k2 in range(ck):
                        for k3 in range(ck):
                            print(i,j,k, k1,k2,k3)
                            out2[i,j,k]+=w[i,k1,k2,k3] * data[k1, h_in_+k2, w_in_+k3]
    print(out.detach() - out2)
    import pdb;pdb.set_trace()

def sanity_test_maxpool2d():
    cout,hout,wout = 1,2,2
    cin, hin, win  = 1,4,4
    ck = 2
    stride = 2

    import pdb;pdb.set_trace()
    #data = torch.arange(cin*hin*win).reshape(cin,hin,win).float()
    data = torch.randn(cin,hin,win)
    out  = F.max_pool2d(data[None,:,:,:], ck).squeeze(dim=0) # 2x2x2
    
    out2 = torch.zeros([cout, hout, wout])
    for i in range(cout):
        for j in range(hout):
            h_in_ = j * stride
            for k in range(wout):
                w_in_ = k * stride
                out2[i,j,k]= -100000
                for k1 in range(ck):
                    for k2 in range(ck):
                        print(i,j,k, k1,k2)
                        out2[i,j,k] = max(out2[i,j,k], data[i, h_in_+k1, w_in_+k2])

    print(out.detach() - out2)
    import pdb;pdb.set_trace()
    
if __name__ == '__main__':
    #sanity_test_conv()
    #sanity_test_maxpool2d()
    lenet = LeNet()
    import pdb;pdb.set_trace()
