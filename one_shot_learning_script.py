import sys
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import tqdm
import pandas as pd
from matching_network import MatchingNetwork
from omniglot_setup import OmniglotNShotDataset
from miniImageNet_setup import MiniImageNetNShotDataset
from cifar_setup import CifarNShotDataset

class FewShot():
    def MetaTraining(self,data,total_train_batches,matchingNet,optimizer,use_cuda,mn_one_shot=False):
        total_c_loss = 0.0
        total_accuracy = 0.0
        if torch.cuda.is_available() & use_cuda:
            matchingNet.cuda()

        for i in range(total_train_batches):
            if mn_one_shot:
                x_support_set, y_support_set, x_target, y_target = data.get_test_batch(False)
            else:
                x_support_set, y_support_set, x_target, y_target = data.get_train_batch(False)
            x_support_set = Variable(torch.from_numpy(x_support_set)).float()
            y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
            x_target = Variable(torch.from_numpy(x_target)).float()
            y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

            y_support_set = y_support_set.unsqueeze(2)
            sequence_length = y_support_set.size()[1]
            batch_size = y_support_set.size()[0]
            y_support_set_one_hot = Variable(torch.zeros(batch_size, sequence_length, data.classes_per_set).scatter_(2,y_support_set.data,1), requires_grad=False)
            
            size = x_support_set.size()
            x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
            x_target = x_target.permute(0, 3, 1, 2)
            if torch.cuda.is_available() & use_cuda:
                acc, c_loss = matchingNet(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(),y_target.cuda())
            else:
                acc, c_loss = matchingNet(x_support_set, y_support_set_one_hot, x_target, y_target)

            optimizer.zero_grad()
            c_loss.backward()
            optimizer.step()

            total_c_loss += c_loss.item()
            total_accuracy += acc.item()

        total_c_loss = total_c_loss / total_train_batches
        total_accuracy = total_accuracy / total_train_batches
        return total_c_loss, total_accuracy

    def MetaValidate(self,data,total_val_batches,matchingNet,scheduler,use_cuda,validate):
        total_c_loss = 0.0
        total_accuracy = 0.0

        for i in range(total_val_batches):
            if validate:
                s_x, s_y, target_x, target_y = data.get_val_batch(False)
            else:
                s_x, s_y, target_x, target_y = data.get_test_batch(False)
            s_x = Variable(torch.from_numpy(s_x)).float()
            s_y = Variable(torch.from_numpy(s_y), requires_grad=False).long()
            target_x = Variable(torch.from_numpy(target_x)).float()
            target_y = Variable(torch.from_numpy(target_y), requires_grad=False).squeeze().long()

            s_y = s_y.unsqueeze(2)
            sequence_length = s_y.size()[1]
            batch_size = s_y.size()[0]
            s_y_oneHot = Variable(torch.zeros(batch_size, sequence_length, data.classes_per_set).scatter_(2,s_y.data,1), requires_grad=False)

            size = s_x.size()
            s_x = s_x.permute(0, 1, 4, 2, 3)
            target_x = target_x.permute(0, 3, 1, 2)
            if torch.cuda.is_available() & use_cuda:
                acc, c_loss = matchingNet(s_x.cuda(), s_y_oneHot.cuda(), target_x.cuda(),target_y.cuda())
            else:
                acc, c_loss = matchingNet(s_x, s_y_oneHot, target_x, target_y)


            total_c_loss += c_loss.item()
            total_accuracy += acc.item()


        total_c_loss = total_c_loss / total_val_batches
        total_accuracy = total_accuracy / total_val_batches
        scheduler.step(total_c_loss)
        return total_c_loss, total_accuracy

    def PreTraining(self,data,model,optimizer,total_train_batches,use_cuda):
        total_c_loss = 0.0
        total_accuracy = 0.0
        if torch.cuda.is_available() & use_cuda:
            model = model.cuda()

        for i in range(total_train_batches):
            train_samples, train_labels = data.get_normal_train_batch(128)
            train_samples = Variable(torch.from_numpy(train_samples)).float()
            train_labels = Variable(torch.from_numpy(train_labels), requires_grad=False).long()

            train_labels = train_labels.unsqueeze(1)
            train_labels_set_one_hot = Variable(torch.zeros(128,data.nClasses).scatter_(1,train_labels.data,1), requires_grad=False)
            
            size = train_samples.size()
            train_samples = train_samples.permute(0, 3, 1, 2)
            if torch.cuda.is_available() & use_cuda:
                pred = model(train_samples.cuda())
                train_labels = train_labels.cuda()
                train_labels_set_one_hot = train_labels_set_one_hot.cuda()
            else:
                pred = model(train_samples)
                
            values, indices = pred.max(1)
            accuracy = torch.mean((indices.squeeze() == train_labels.squeeze(1)).float())
            crossentropy_loss = F.cross_entropy(pred, train_labels.squeeze(1).long())


            optimizer.zero_grad()
            crossentropy_loss.backward()
            optimizer.step()

            total_c_loss += crossentropy_loss.item()
            total_accuracy += accuracy.item()

        total_c_loss = total_c_loss / total_train_batches
        total_accuracy = total_accuracy / total_train_batches
        return total_c_loss, total_accuracy