import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.io import loadmat
import torch.optim as optim
import torch.utils.data as data_utils
from ConsenNet_utils import prepare_twoDoamin_data, prepare_averaged_template_data, prepare_M_list
import logging
from collections import OrderedDict
## parameters
# channels: Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
permutation = [47, 53, 54, 55, 56, 57, 60, 61, 62]
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ContrastiveLoss(nn.Module):
    def __init__(self, n_classes):
        super(ContrastiveLoss, self).__init__()
        self.cl = n_classes

    def forward(self, input, label, temperature):  # input: [bs, cl]   label: [bs]
        pos_mask = F.one_hot(label, num_classes=self.cl)  # [bs, cl]
        pos_sim = torch.sum(input * pos_mask, dim=1)  # [bs]
        pos_sim = torch.exp(pos_sim/temperature)  # [bs]

        neg_mask = (torch.ones_like(pos_mask) - pos_mask).bool()
        neg_sim = input.masked_select(neg_mask).view(-1, self.cl-1)  # [bs, cl-1]
        neg_sim = torch.exp(neg_sim/temperature)  # [bs, cl-1]
        neg_sim = torch.sum(neg_sim, dim=1)  # [bs]

        return (-torch.log(pos_sim / neg_sim)).sum()

class multi_ch_Corr(nn.Module):
    def __init__(self, args, num, **kwargs):
        self.tw = 145
        self.cl = args.y_dim
        self.num = num
        self.corr = None
        super(multi_ch_Corr, self).__init__(**kwargs)
    def forward(self, input, **kwargs):
        x = input[0]  # [bs, tw * kernel_size_2]  signal  compX[bs,120*145]
        bs = x.shape[0]
        x_ = torch.reshape(x, (-1, self.num, self.tw, 1))
        x_ = torch.transpose(x_, 1, 2) # [bs, 145, 120, 1]
        # x_ = torch.reshape(x, (-1, self.tw, self.num, 1))  # [bs, tw, kernel_size_2, 1]

        t = input[1]  # [cl, 1, tw * kernel_size_2, cl] reference
        t_ = torch.reshape(t, (-1, self.num, self.tw, 1))
        t_ = torch.transpose(t_, 1, 2)  # [cl, 145, 120, 1]
        t_ = torch.transpose(t_, 0, 3)  # [1, 145, 120, cl]
        t_ = t_.repeat(bs, 1, 1, 1)    # [cl, 145, 120, cl]
        # t_ = torch.reshape(t, (-1, self.tw, self.num, self.cl))  # [bs, tw, kernel_size_2, cl]

        corr_xt = torch.sum(x_*t_, dim=1)  # [bs, kernel_size_2, cl]
        corr_xx = torch.sum(x_*x_, dim=1)  # [bs, kernel_size_2, 1]
        corr_tt = torch.sum(t_*t_, dim=1)  # [bs, kernel_size_2, cl]
        self.corr = corr_xt/torch.sqrt(corr_tt)/torch.sqrt(corr_xx)  # [bs, kernel_size_2, cl]
        self.out = self.corr  # [bs, kernel_size_2, cl]
        self.out = torch.mean(self.out, dim=1)  # [bs, cl]
        return self.out

class myNet(nn.Module):
    def __init__(self,
                 n_features,
                 args, source_data,
                 band_kernel=9, pooling_kernel=2,
                 eps=1e-8, dropout1=0.5, dropout2=0.95):
        super().__init__()
        self.n_features = n_features
        self.n_channels = args.ch
        self.n_samples = args.tw
        self.n_classes = args.y_dim
        self.eps = torch.tensor(eps)

        self.alpha = args.alpha

        self.feature_extractor_compX_f = nn.Sequential(OrderedDict([
            ('spatial_layer', nn.Conv2d(
                1, n_features, (self.n_channels, 1),
                stride=(1, 1), padding=(0, 0), bias=False)),
            ('bs1', nn.BatchNorm2d(n_features)),  #
            ('relu1', nn.ReLU()),  #
            ('drop1', nn.Dropout(dropout1)),  #
            ('temporal_layer1', nn.Conv2d(
                n_features, n_features, (1, pooling_kernel),
                stride=(1, pooling_kernel), padding=(0, 0), bias=False)),
            ('bn_layer', nn.BatchNorm2d(n_features)),
            ('relu2', nn.ReLU()),
            ('drop2', nn.Dropout(dropout1)),  #
            ('temporal_layer2', nn.Conv2d(
                n_features, n_features, (1, band_kernel),
                stride=(1, 1), padding=(0, int((band_kernel-1)/2)), bias=False)),
        ]))

        self.spatial_feature_extractor = nn.Sequential(OrderedDict([
            ('spatial_layer', nn.Conv2d(
                1, n_features, (self.n_channels, 1),
                stride=(1, 1), padding=(0, 0), bias=False)),
            ('bs1', nn.BatchNorm2d(n_features)),  #
            ('relu0', nn.ReLU()),  #
            ('drop0', nn.Dropout(dropout1)),  #
        ]))

        self.real_imag_feature_extractor = nn.Sequential(OrderedDict([
            ('real_imag_layer', nn.Conv2d(
                n_features, 2*n_features, (2, 1),
                stride=(1, 1), padding=(0, 0), bias=False)),
            ('bs1', nn.BatchNorm2d(2*n_features)),  #
            ('relu1', nn.ReLU()),  #
            ('drop1', nn.Dropout(dropout1)),  #
        ]))

        self.freq_feature_extractor = nn.Sequential(OrderedDict([
            ('temporal_layer1', nn.Conv2d(
                2*n_features, n_features, (1, pooling_kernel),
                stride=(1, pooling_kernel), padding=(0, 0), bias=False)),
            ('bn_layer', nn.BatchNorm2d(n_features)),
            ('relu2', nn.ReLU()),
            ('drop2', nn.Dropout(dropout1)),  #
            ('temporal_layer2', nn.Conv2d(
                n_features, n_features, (1, band_kernel),
                stride=(1, 1), padding=(0, int((band_kernel - 1) / 2)), bias=False)),
        ]))

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc_drop = nn.Dropout(dropout2)


        with torch.no_grad():
            compX = torch.zeros(1, 1, 9, 291)
            compX = self.feature_extractor_compX_f(compX)
            compX_out = self.flatten(compX)

        self.bn01 = nn.BatchNorm2d(n_features)
        self.fc_layer1 = nn.Linear(compX_out.shape[-1], self.n_classes)

        self.instance_norm_2 = nn.InstanceNorm2d(2)

    def forward(self, compX, y): # compX [bs, 2, ch, 291]
        compX = self.instance_norm_2(compX)  # [bs, 2, ch, 291]
        # compX = self.feature_extractor_compX_f(compX)    # [bs, 120, 1, 281]
        compX = compX.reshape(-1, 1, self.n_channels, 291) # [bs*2, 1, ch, 291]
        compX = self.spatial_feature_extractor(compX)  # [bs*2, 120, 1, 291]

        compX = compX.reshape(-1, 2, self.n_features, 291)   # [bs, 2, 120, 291]
        compX = torch.transpose(compX, 1, 2)    # [bs, 120, 2, 291]
        compX = self.real_imag_feature_extractor(compX)   # [bs, 120, 1, 291]
        compX = self.freq_feature_extractor(compX)

        compX = self.bn01(compX)

        compX = self.flatten(compX)

        out_f = self.relu(compX)
        out_f = self.fc_drop(out_f)
        out_f = self.fc_layer1(out_f)
        return out_f, compX


class myNet_ts(nn.Module):
    def __init__(self,
                 n_features,
                 args, source_data,
                 band_kernel=9, pooling_kernel=2,
                dropout1=0.5, dropout2=0.95):
        super().__init__()

        self.teacher = myNet(n_features, args, source_data,
                band_kernel=band_kernel, pooling_kernel=pooling_kernel,
                dropout1=dropout1, dropout2=dropout2)
        self.student = myNet(n_features, args, source_data,
                             band_kernel=band_kernel, pooling_kernel=pooling_kernel,
                             dropout1=dropout1, dropout2=dropout2)
        Template = torch.as_tensor(source_data, dtype=torch.float)
        self.register_buffer('Template', Template)  # (35, cl, 2, 281, ch)

        self.corr = multi_ch_Corr(args=args, num=n_features)
        self.t = args.temperature
        self.n_classes = args.y_dim
        self.alpha = args.alpha


    def forward(self, compX, y): # compX [bs, 2, ch, 291]
        s = self.Template[:]  # (cl, 2, 281, ch)
        s = torch.transpose(s, 2, 3)  # [bs, 2, ch, 281]

        _, s = self.teacher(s, y)
        s = s.detach()

        out_f, compX = self.student(compX, y)

        corr = self.corr([compX, s])  # [bs, cl]
        return out_f, corr

    def loss_function(self, compx, y):  # supervised
        out_f,  corr = self.forward(compx, y)
        bs = out_f.size()[0]
        # calculate MSE loss
        CE_f = F.cross_entropy(out_f, y, reduction='sum')
        criterion_Contra = ContrastiveLoss(self.n_classes)
        contrastive_loss = criterion_Contra(corr, y, temperature=self.t)

        loss = CE_f + contrastive_loss * self.alpha
        return loss, CE_f, contrastive_loss, out_f


def train(train_loader, model, optimizer):
    model.train()
    train_loss = 0
    train_contra_loss = 0
    acc_f = 0
    for batch_idx, (x, compx, y) in enumerate(train_loader):
        # To device
        x, compx, y = x.to(device), compx.to(device), y.to(device)
        optimizer.zero_grad()
        loss, CE_loss, contra_loss, class_out_f = model.loss_function(compx, y)
        loss.backward()
        optimizer.step()
        train_loss += loss
        train_contra_loss += contra_loss
        correct_f = get_accuracy(class_out_f, y)
        acc_f += correct_f
    train_loss /= len(train_loader.dataset)
    train_contra_loss /= len(train_loader.dataset)
    acc_f /= len(train_loader.dataset)
    return train_loss, train_contra_loss, acc_f*100

def ttest(test_loader, model, optimizer):
    model.eval()
    test_loss = 0
    test_contra_loss = 0
    acc_f = 0
    with torch.no_grad():
        for batch_idx, (x, compx, y) in enumerate(test_loader):
            # To device
            x, compx, y = x.to(device),compx.to(device), y.to(device)
            optimizer.zero_grad()
            loss, CE_loss, contra_loss, class_out_f= model.loss_function(compx, y)
            test_loss += loss
            test_contra_loss += contra_loss
            correct_f = get_accuracy(class_out_f, y)
            acc_f += correct_f

    test_loss /= len(test_loader.dataset)
    test_contra_loss /= len(test_loader.dataset)
    acc_f /= len(test_loader.dataset)
    return test_loss, test_contra_loss, acc_f*100

def get_accuracy(class_out, label):
    _, predicted = torch.max(class_out.data, 1)
    correct = (predicted == label).sum().item()
    return correct


def main():
    vali_table = loadmat('./Benchmark_validation_set.mat')['data']  # [35, 3]
    FFT_PARAMS = {
        'resolution': 0.2,
        'start_frequency': 6,
        'end_frequency': 64,
        'sampling_rate': 250
    }

    for tw in [125]:
        # Training settings
        parser = argparse.ArgumentParser(description='ConsenNet')  # 创建解析器
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')  # 添加参数
        parser.add_argument('--seed', type=int, default=0,
                            help='random seed (default: 1)')
        parser.add_argument('--batch-size', type=int, default=100,
                            help='input batch size for training (default: 200)')
        parser.add_argument('--epochs', type=int, default=2500,
                            help='number of epochs to train (default: 2500)')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='learning rate (default: 0.0001)')

        parser.add_argument('--alpha', type=float, default=1.0,
                            help='multiplier for y classification loss')
        parser.add_argument('--y-dim', type=int, default=40,
                            help='number of classes')
        parser.add_argument('--tw', type=int, default=tw,
                            help='length of signal')
        parser.add_argument('--ch', type=int, default=9,
                            help='number of channels')
        parser.add_argument('--train_patience', type=float, default=100,
                            help='Training patience for validation loss.')
        parser.add_argument('--temperature', type=float, default=1.0,
                            help='Temperature for contrastive loss')

        args = parser.parse_args()  # 解析参数
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

        # Set seed
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False  # if flag=True 可大大提高程序运行效率
        np.random.seed(args.seed)

        for subj in range(1, 36):
            vali_subj_list = vali_table[subj - 1, :]  # [3]

            train_subj_list = [i for i in range(1, 36)]
            train_subj_list.remove(vali_subj_list[0])
            train_subj_list.remove(vali_subj_list[1])
            train_subj_list.remove(vali_subj_list[2])
            train_subj_list.remove(subj)

            train_run = [0, 1, 2, 3, 4, 5]
            vali_run = [0, 1, 2, 3, 4, 5]

            # Load training data
            train_loader = data_utils.DataLoader(
                prepare_twoDoamin_data(subj_list=train_subj_list, blocks=train_run, tw=args.tw,
                                         M_list_all=[], M_list_20=[], M_list_10=[], M_list_5=[],FFT_PARAMS=FFT_PARAMS),
                batch_size=args.batch_size,
                shuffle=True, **kwargs)

            # Load validation data
            vali_loader = data_utils.DataLoader(
                prepare_twoDoamin_data(subj_list=vali_subj_list, blocks=vali_run, tw=args.tw,
                                       M_list_all=[], M_list_20=[], M_list_10=[],
                                       M_list_5=[], FFT_PARAMS=FFT_PARAMS),
                batch_size=args.batch_size,
                shuffle=True, **kwargs)


            source_data, complex_source = prepare_averaged_template_data(Bench_subj=train_subj_list,  tw=args.tw, FFT_PARAMS=FFT_PARAMS)

            load_dir = './PreTrain_stage1/tw'+ str(int(tw)) +'/preTrain_for_subj' + str(subj) + '.pth'
            save_dir = './PreTrain_stage2/tw'+ str(int(tw)) +'/preTrain_for_subj' + str(subj) + '.pth'

            print('The best model will be saved under path :  ' + save_dir)

            model = myNet_ts(
                120, args, complex_source,
                band_kernel=9, pooling_kernel=2,
                dropout1=0.5, dropout2=0.95)
            model.teacher.load_state_dict(torch.load(load_dir))
            model.student.load_state_dict(torch.load(load_dir))
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
            # training loop
            print('\nStart training:', args)
            max_vali_acc = 0
            patient=0
            for epoch in range(1, args.epochs + 1):
                start_time = time.time()
                # train
                train_total_loss, train_contra_loss, train_acc_f = train(train_loader, model, optimizer)
                validate_total_loss, validate_contra_loss, validate_acc_f = ttest(vali_loader, model, optimizer)

                # store the loss and validation/testing accuracies in the logfile
                str_print = "{} epoch --- train total loss {:.4f}".format(epoch, train_total_loss)
                str_print += " , contra loss {:.4f}".format(train_contra_loss)
                str_print += ", acc_f {:.2f}".format(train_acc_f)

                str_print += " --- validate total loss {:.4f}".format(validate_total_loss)
                str_print += ", contra loss {:.4f}".format(validate_contra_loss)
                str_print += ", acc_f {:.2f}".format(validate_acc_f)

                str_print += ", time {:.2f}".format((time.time() - start_time))

                print(str_print)

                # save best model's parameter
                if validate_acc_f > max_vali_acc:
                    patient = 0
                    max_vali_acc = validate_acc_f
                    torch.save(model.state_dict(), save_dir)
                    print('Model saved!')
                else:
                    patient = patient + 1
                if patient > args.train_patience:
                    break

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    with torch.cuda.device(0):
        main()
