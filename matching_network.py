import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

class OmniglotClassifier(nn.Module):
    def __init__(self, layer_size, nClasses = 0, num_channels = 1, keep_prob = 0.0, image_size = 28, freeze = False):
        super(OmniglotClassifier, self).__init__()
        self.layer1 = self.convLayer(num_channels, layer_size, keep_prob)
        self.layer2 = self.convLayer(layer_size, layer_size, keep_prob)
        self.layer3 = self.convLayer(layer_size, layer_size, keep_prob)
        self.layer4 = self.convLayer(layer_size, layer_size, keep_prob)

        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size
        if nClasses>0: # We want a linear
            self.useClassification = True
            self.layer5 = nn.Linear(self.outSize,nClasses)
            self.outSize = nClasses
        else:
            self.useClassification = False
        
    def convLayer(self, in_channels, out_channels, keep_prob):
        cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(keep_prob)
        )
        return cnn_seq
    
    def freeze(self):
        for name, parameter in self.named_parameters():
            parameter.requries_grad = False

    def forward(self, image_input):
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size()[0], -1)
        if self.useClassification:
            x = self.layer5(x)
        return x

class CifarClassifier(nn.Module):
    def __init__(self, layer_size, nClasses = 0, num_channels = 1, keep_prob = 0.0, image_size = 32, freeze = False):
        super(CifarClassifier, self).__init__()
        self.layer1 = self.convLayer(num_channels, layer_size, keep_prob)
        self.layer2 = self.convLayer(layer_size, layer_size, keep_prob)
        self.layer3 = self.convLayer(layer_size, layer_size, keep_prob)
        self.layer4 = self.convLayer(layer_size, layer_size, keep_prob)
        self.layer5 = self.convLayer(layer_size, layer_size, keep_prob)

        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size
        if nClasses>0: # We want a linear
            self.useClassification = True
            self.layer6 = nn.Linear(self.outSize,nClasses)
            self.outSize = nClasses
        else:
            self.useClassification = False
        
    def convLayer(self, in_channels, out_channels, keep_prob):
        cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(keep_prob)
        )
        return cnn_seq
    
    def freeze(self):
        for name, parameter in self.named_parameters():
            parameter.requries_grad = False

    def forward(self, image_input):
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size()[0], -1)
        if self.useClassification:
            x = self.layer6(x)
        return x

class miniImageNetClassifier(nn.Module):
    def __init__(self, layer_size, nClasses = 0, num_channels = 1, keep_prob = 0.0, image_size = 28, freeze = False):
        super(miniImageNetClassifier, self).__init__()
        self.layer1 = self.convLayer(num_channels, layer_size, keep_prob)
        self.layer2 = self.convLayer(layer_size, layer_size, keep_prob)
        self.layer3 = self.convLayer(layer_size, layer_size, keep_prob)
        self.layer4 = self.convLayer(layer_size, layer_size, keep_prob)
        self.layer5 = self.convLayer(layer_size, layer_size, keep_prob)
        self.layer6 = self.convLayer(layer_size, layer_size, keep_prob)

        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size
        if nClasses>0: # We want a linear
            self.useClassification = True
            self.layer7 = nn.Linear(self.outSize,nClasses)
            self.outSize = nClasses
        else:
            self.useClassification = False
            
            
        
    def convLayer(self, in_channels, out_channels, keep_prob):
        cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(keep_prob)
        )
        return cnn_seq
    
    def freeze(self):
        for name, parameter in self.named_parameters():
            parameter.requries_grad = False

    def forward(self, image_input):
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size()[0], -1)
        if self.useClassification:
            x = self.layer7(x)
        return x

class CosineAttentionClassifier(nn.Module):
    def __init__(self):
        super(CosineAttentionClassifier, self).__init__()

    def forward(self, s, target, s_label):
        eps = 1e-10
        similarity = []
        m_target = torch.sum(torch.pow(target, 2), 1)
        r_target = m_target.clamp(eps, float("inf")).rsqrt()
        for x_i in s:
            m_xi = torch.sum(torch.pow(x_i, 2), 1)
            r_xi = m_xi.clamp(eps, float("inf")).rsqrt()
            dot = target.unsqueeze(1).bmm(x_i.unsqueeze(2)).squeeze()
            c_s = dot * r_target * r_xi
            similarity.append(c_s)
        similarity = torch.stack(similarity)
        
        #softmax = nn.Softmax()
        #softmax_sim = softmax(similarity.t())
        #preds = softmax_sim.unsqueeze(1).bmm(s_label).squeeze()
        
        #return preds
        return similarity.t()

class FCE_g(nn.Module):
    def __init__(self, layer_size, batch_size, vector_dim,use_cuda):
        super(FCE_g, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = layer_size
        self.vector_dim = vector_dim
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=1, hidden_size=self.hidden_size,
                            bidirectional=True)

    def init_hidden(self,use_cuda):
        if use_cuda:
            return (Variable(torch.zeros(2, self.batch_size, self.lstm.hidden_size),requires_grad=True).cuda(),
                    Variable(torch.zeros(2, self.batch_size, self.lstm.hidden_size),requires_grad=True).cuda())
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.lstm.hidden_size),requires_grad=True),
                    Variable(torch.zeros(2, self.batch_size, self.lstm.hidden_size),requires_grad=True))

    def forward(self, inputs):
        hidden = self.init_hidden(self.use_cuda)
        output, _ = self.lstm(inputs,hidden)
        outputs = torch.add(inputs,output)
        return outputs

class FCE_f(nn.Module):
    def __init__(self,read_steps,layer_size,batch_size,vector_dim,use_cuda):
        super(FCE_f,self).__init__()
        self.read_steps = read_steps
        self.batch_size = batch_size
        self.hidden_size = layer_size
        self.vector_dim = vector_dim
        self.use_cuda = use_cuda
        self.lstmRead = nn.LSTMCell(input_size=self.vector_dim,hidden_size=self.hidden_size)
        self.softmax = nn.Softmax(dim=0)
        
    def init_hidden(self,use_cuda):
        if use_cuda:
            return (Variable(torch.zeros(self.batch_size, self.lstmRead.hidden_size),requires_grad=True).cuda(),
                    Variable(torch.zeros(self.batch_size, self.lstmRead.hidden_size),requires_grad=True).cuda())
        else:
            return (Variable(torch.zeros(self.batch_size, self.lstmRead.hidden_size),requires_grad=True),
                    Variable(torch.zeros(self.batch_size, self.lstmRead.hidden_size),requires_grad=True))
        
    def forward(self, inputs, memory):
        prev_hc = self.init_hidden(self.use_cuda)
        for step in range(self.read_steps):
            hidden_cap, cell = self.lstmRead(inputs, prev_hc)
            h_k = torch.add(hidden_cap,inputs)
            content_based_attention = self.softmax(torch.mul(prev_hc[0],memory))
            r_k = torch.sum(torch.mul(content_based_attention,memory),axis=0)
            
            prev_hc = tuple((torch.add(h_k,r_k),cell))
            
        return h_k

class MatchingNetwork(nn.Module):
    def __init__(self, keep_prob=0.0, batch_size=32, num_channels=1, fce=False, image_size=28, use_cuda=True, model = 0):
        super(MatchingNetwork, self).__init__()
        '''Parameters'''
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.fce = fce
        self.image_size = image_size
        
        '''Networks'''
        if model == 0:
            self.embedding = OmniglotClassifier(layer_size=64,nClasses=0,num_channels=num_channels,keep_prob=keep_prob,image_size=image_size)
        elif model == 1:
            self.embedding = miniImageNetClassifier(layer_size=64,nClasses=0,num_channels=num_channels,keep_prob=keep_prob,image_size=image_size)
        else:
            self.embedding = CifarClassifier(layer_size=64,nClasses=0,num_channels=num_channels,keep_prob=keep_prob,image_size=image_size)
        self.classifier = CosineAttentionClassifier()
        
        if self.fce:
            self.FCE_f = FCE_f(10,layer_size=64, batch_size=self.batch_size, vector_dim=self.embedding.outSize,use_cuda=use_cuda)
            self.FCE_g = FCE_g(layer_size=32, batch_size=self.batch_size, vector_dim=self.embedding.outSize,use_cuda=use_cuda)

    def forward(self, s, s_y, target, target_y):
        embedded_xi = []
        for i in np.arange(s.size(1)):
            x_i = self.embedding(s[:, i, :, :])
            embedded_xi.append(x_i)
        support = torch.stack(embedded_xi)

        sample = self.embedding(target)

        if self.fce:
            FCE_support = self.FCE_g(support)
            FCE_sample = self.FCE_f(sample, FCE_support)
            preds = self.classifier(s = FCE_support, target = FCE_sample, s_label=s_y)
            
        else:
            preds = self.classifier(s = support, target = sample, s_label=s_y)
        
        values, indices = preds.max(1)
        accuracy = torch.mean((indices.squeeze() == target_y).float())
        #crossentropy_loss = F.nll_loss(preds, target_y.long())
        crossentropy_loss = F.cross_entropy(preds, target_y.long())
        

        return accuracy, crossentropy_loss

