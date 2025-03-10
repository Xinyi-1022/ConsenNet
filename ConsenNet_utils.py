from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import cheb1ord, filtfilt, cheby1

Fs = 250. # sampling freq
def prepare_M_list(all_subj_list, N20=0, N10=0, N5=70):
    """
    :param all_subj_list: N subjects used for training
    :param N20: number of sampling times for 20 subjects averaging
    :param N10: number of sampling times for 10 subjects averaging
    :param N5: number of sampling times for 5 subjects averaging ]
    """
    N = len(all_subj_list)
    M_list_all = np.array([], dtype=np.int32).reshape(0, N)
    M_list_20 = np.array([], dtype=np.int32).reshape(0, 20)
    M_list_10 = np.array([], dtype=np.int32).reshape(0, 10)
    M_list_5 = np.array([], dtype=np.int32).reshape(0, 5)
    for i in range(1):
        rand = np.random.choice(all_subj_list, N, replace=False).reshape(1, -1)  # replace=False 无放回抽样
        M_list_all = np.append(M_list_all, rand, axis=0)
    for i in range(N20):
        rand = np.random.choice(all_subj_list, 20, replace=False).reshape(1, -1)  # replace=False 无放回抽样
        M_list_20 = np.append(M_list_20, rand, axis=0)  # [?,tw,ch], ?=runs*cl
    for i in range(N10):
        rand = np.random.choice(all_subj_list, 10, replace=False).reshape(1, -1)  # replace=False 无放回抽样
        M_list_10 = np.append(M_list_10, rand, axis=0)  # [?,tw,ch], ?=runs*cl
    for i in range(N5):
        rand = np.random.choice(all_subj_list, 5, replace=False).reshape(1, -1)  # replace=False 无放回抽样
        M_list_5 = np.append(M_list_5, rand, axis=0)  # [?,tw,ch], ?=runs*cl
    return M_list_all, M_list_20, M_list_10, M_list_5

def prepare_averaged_template_data(Bench_subj, tw, FFT_PARAMS):
    # Prepare template in the benchmark dataset
    file_Bench = loadmat('./subj_templates.mat')['data']  # [35, 40, 1250, 9]
    temp_data_Bench = file_Bench[[item - 1 for item in Bench_subj], :, 0:tw, :]  # [Bench_subj, 40, tw, 9]
    print('Template data has been prepared for Benchmark, shape: ',temp_data_Bench.shape)

    temp_data = np.mean(temp_data_Bench, axis=0)  # [40, tw, 9]
    complex_x = complex_spectrum_feature(temp_data, FFT_PARAMS)  # [40, 2, 281, 9]
    print('Template data has been prepared, shape: ', temp_data.shape)

    return temp_data, complex_x

def prepare_generated_data_as(subj_list, blocks, tw, M_list_all, M_list_20, M_list_10, M_list_5, permutation):
    ch = len(permutation)  # # of channels
    N = len(subj_list)
    X_all = np.zeros((35, ch, tw, 40, len(blocks)))
    gen_subj_5 = np.zeros((len(M_list_5), ch, tw, 40, len(blocks)))  # generated data averaged by 5 subjects
    gen_subj_10 = np.zeros((len(M_list_10), ch, tw, 40, len(blocks)))
    gen_subj_20 = np.zeros((len(M_list_20), ch, tw, 40, len(blocks)))
    gen_subj_all = np.zeros((len(M_list_all), ch, tw, 40, len(blocks)))
    for subj in subj_list:
        file = loadmat('../SSVEP_datasets/benchmark/S' + str(subj) + '.mat')['data']
        raw_data = file[permutation, int(0.14 * 250 + 125): int(0.14 * 250 + 125)+tw, :, :]  # [9, tw, 40, 6]
        raw_data = raw_data[:, :, :, blocks]
        X_all[subj-1, :,:,:,:] = raw_data[:]

    for i in range(len(M_list_5)):
        subj_index = M_list_5[i, :]  #[5]
        subj_index = subj_index-1
        gen_subj_5[i, :,:,:] = np.mean(X_all[subj_index, :,:,:,:], axis=0, keepdims=True)

    for i in range(len(M_list_10)):
        subj_index = M_list_10[i, :]  #[10]
        subj_index = subj_index - 1
        gen_subj_10[i, :,:,:] = np.mean(X_all[subj_index, :,:,:,:], axis=0, keepdims=True)

    for i in range(len(M_list_20)):
        subj_index = M_list_20[i, :]  #[20]
        subj_index = subj_index - 1
        gen_subj_20[i, :,:,:] = np.mean(X_all[subj_index, :,:,:,:], axis=0, keepdims=True)

    for i in range(len(M_list_all)):
        subj_index = M_list_all[i, :]  #[31]
        subj_index = subj_index - 1
        gen_subj_all[i, :,:,:] = np.mean(X_all[subj_index, :,:,:,:], axis=0, keepdims=True)

    return X_all, gen_subj_5, gen_subj_10, gen_subj_20, gen_subj_all


class prepare_twoDoamin_data(Dataset):
    def __init__(self, subj_list, blocks, tw, M_list_all, M_list_20, M_list_10, M_list_5, FFT_PARAMS,permutation=[47,53,54,55,56,57,60,61,62]):
        ## build signal and label
        x, complex_x, y = prepare_twoDomain_data_as(subj_list, blocks, tw, M_list_all, M_list_20, M_list_10, M_list_5, FFT_PARAMS,permutation=permutation)  # [?,tw,ch]

        self.y = torch.Tensor(y).type(torch.LongTensor)
        self.x = x.transpose(0, 2, 1)    # [?,ch,tw]
        self.x = self.x.astype('float32')
        # self.complex_x = complex_x.transpose(0,2,1)
        self.complex_x = complex_x.transpose(0, 1, 3, 2)  # [?, 2, ch, 281]
        self.complex_x = self.complex_x.astype('float32')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        signal = self.x[item]
        complex_signal = self.complex_x[item]
        label = self.y[item]
        return signal, complex_signal, label


def prepare_twoDomain_data_as(subj_list, blocks, tw, M_list_all, M_list_20, M_list_10, M_list_5,
    FFT_PARAMS, cl=40,permutation=[47,53,54,55,56,57,60,61,62]):
    ch = len(permutation) # # of channels
    x = np.array([],dtype=np.float32).reshape(0,tw,ch) # data
    y = np.zeros([0],dtype=np.int32) # true label
    X_all, gen_subj_5, gen_subj_10, gen_subj_20, gen_subj_all = prepare_generated_data_as(subj_list, blocks, tw, M_list_all, M_list_20, M_list_10, M_list_5, permutation)

    if len(M_list_all)!=0:
        for subj in range(len(gen_subj_5)):
            file = gen_subj_5[subj,:,:,:,:]
            for run_idx in blocks:
                for freq_idx in range(cl):
                    raw_data = file[:,:,freq_idx,run_idx].T
                    n_samples = 1
                    _x = np.zeros([n_samples,tw,ch],dtype=np.float32)
                    _y = np.ones([n_samples],dtype=np.int32) * freq_idx

                    _x[0,:,:] = raw_data[0:tw,:]

                    x = np.append(x,_x,axis=0) # [?,tw,ch], ?=runs*cl
                    y = np.append(y,_y)        # [?,1]
        print('Data averaged by 5 subjects has been prepared! Generated ', len(gen_subj_5),' new subjects.')
        for subj in range(len(gen_subj_10)):
            file = gen_subj_10[subj,:,:,:,:]
            for run_idx in blocks:
                for freq_idx in range(cl):
                    raw_data = file[:,:,freq_idx,run_idx].T
                    n_samples = 1
                    _x = np.zeros([n_samples,tw,ch],dtype=np.float32)
                    _y = np.ones([n_samples],dtype=np.int32) * freq_idx

                    _x[0,:,:] = raw_data[0:tw,:]

                    x = np.append(x,_x,axis=0) # [?,tw,ch], ?=runs*cl
                    y = np.append(y,_y)        # [?,1]
        print('Data averaged by 10 subjects has been prepared! Generated ', len(gen_subj_10), ' new subjects.')
        for subj in range(len(gen_subj_20)):
            file = gen_subj_20[subj,:,:,:,:]
            for run_idx in blocks:
                for freq_idx in range(cl):
                    raw_data = file[:,:,freq_idx,run_idx].T
                    n_samples = 1
                    _x = np.zeros([n_samples,tw,ch],dtype=np.float32)
                    _y = np.ones([n_samples],dtype=np.int32) * freq_idx

                    _x[0,:,:] = raw_data[0:tw,:]

                    x = np.append(x,_x,axis=0) # [?,tw,ch], ?=runs*cl
                    y = np.append(y,_y)        # [?,1]
        print('Data averaged by 20 subjects has been prepared! Generated ', len(gen_subj_20), ' new subjects.')
        for subj in range(len(gen_subj_all)):
            file = gen_subj_all[subj,:,:,:,:]
            for run_idx in blocks:
                for freq_idx in range(cl):
                    raw_data = file[:,:,freq_idx,run_idx].T
                    n_samples = 1
                    _x = np.zeros([n_samples,tw,ch],dtype=np.float32)
                    _y = np.ones([n_samples],dtype=np.int32) * freq_idx

                    _x[0,:,:] = raw_data[0:tw,:]

                    x = np.append(x,_x,axis=0) # [?,tw,ch], ?=runs*cl
                    y = np.append(y,_y)        # [?,1]
        print('Data averaged by all subjects has been prepared! Generated ', len(gen_subj_all), ' new subjects.')

    for subj in subj_list:
        file = X_all[subj-1, :, :, :, :]
        for run_idx in range(len(blocks)):
            for freq_idx in range(cl):
                raw_data = file[:, :, freq_idx, run_idx].T
                n_samples = 1
                _x = np.zeros([n_samples, tw, ch], dtype=np.float32)
                _y = np.ones([n_samples], dtype=np.int32) * freq_idx

                _x[0, :, :] = raw_data[0:tw, :]

                x = np.append(x, _x, axis=0)  # [?,tw,ch], ?=runs*cl
                y = np.append(y, _y)  # [?,1]
    print('Data for each single subject has been prepared!')

    x = filter(x)
    complex_x = complex_spectrum_feature(x, FFT_PARAMS)

    print('S'+str(subj)+'|x',x.shape)
    return x, complex_x, y
def complex_spectrum_feature(x, FFT_PARAMS): # x [?, tw, ch]
    num_trial = x.shape[0]
    tw = x.shape[1]
    num_ch = x.shape[2]
    NFFT = round(FFT_PARAMS['sampling_rate'] / FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency'] / FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency'] / FFT_PARAMS['resolution'])) + 1

    features_data = np.zeros((x.shape[0], 2, (1 * (fft_index_end - fft_index_start)), x.shape[2]))
    for trial in range(0, num_trial):
        for ch in range(0, num_ch):
            temp_FFT = np.fft.fft(x[trial, :, ch], NFFT) / tw * 2
            real_part = np.real(temp_FFT)
            imag_part = np.imag(temp_FFT)
            features_data[trial, 0, :, ch] = real_part[fft_index_start:fft_index_end, ]
            features_data[trial, 1, :, ch] = imag_part[fft_index_start:fft_index_end, ]
    return features_data


## prepossing by Chebyshev Type I filter
def filter(x):
    nyq = 0.5 * Fs
    Wp = [6/nyq, 90/nyq];
    Ws = [4/nyq, 100/nyq];
    N, Wn=cheb1ord(Wp, Ws, 3, 40);
    b, a = cheby1(N, 0.5, Wn,'bandpass');
    # --------------
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            _x = x[i,:,j]
            x[i,:,j] = filtfilt(b,a,_x) # apply filter

    return x

