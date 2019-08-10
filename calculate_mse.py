from config import ConfigArgs as args
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
from model import Text2Mel, SSRN
from data import SpeechDataset, t2m_collate_fn, load_vocab
import utils
from scipy.io.wavfile import write
import time


def calculate_MSE(t2m, ssrn, data_loader, batch_size=100):
    '''
    DCTTS Architecture
    Text --> Text2Mel --> SSRN --> Wav file
    '''

    with torch.no_grad():
        mse = 0.
        length = 0
        for step, (texts, mels, extras) in enumerate(data_loader):
            first_frames = torch.zeros([mels.shape[0], 1, args.n_mels]).to(DEVICE) # (N, Ty/r, n_mels)
            texts, mels = texts.to(DEVICE), mels.to(DEVICE)
            prev_mels = torch.cat((first_frames, mels[:, :-1, :]), 1)
            mels_hat, A, _ = t2m(texts, prev_mels)  # mels_hat: (N, Ty/r, n_mels), A: (N, Tx, Ty/r)

            mean_error = ((mels_hat - mels)**2).mean()
            mse += mean_error

            length+= mels.size()[0]       
        mse = mse.cpu().numpy()


    return mse/length, length


        

def main(mode):


    t2m = Text2Mel().to(DEVICE)
    ssrn = SSRN().to(DEVICE)


    if mode == "train":
        dataset = SpeechDataset(args.data_path, args.meta_train, "Text2Mel", mem_mode=args.mem_mode)
    elif mode=="test":
        dataset = SpeechDataset(args.data_path, args.meta_test, "Text2Mel", mem_mode=args.mem_mode)
    elif mode=="eval":
        dataset = SpeechDataset(args.data_path, args.meta_eval, "Text2Mel", mem_mode=args.mem_mode)

    else:
        print('[ERROR] Please set correct type: TRAIN or TEST!' )
        exit(0)


    data_loader = DataLoader(dataset=dataset, batch_size=args.mse_batch,
                             shuffle=False, collate_fn=t2m_collate_fn, pin_memory=True)


    
    ckpt = pd.read_csv(os.path.join(args.logdir, t2m.name, 'ckpt.csv'), sep=',', header=None)
    ckpt.columns = ['models', 'loss']
    ckpt = ckpt.sort_values(by='loss', ascending=True)
    state = torch.load(os.path.join(args.logdir, t2m.name, ckpt.models.loc[0]))
    t2m.load_state_dict(state['model'])
    args.global_step = state['global_step']

    ckpt = pd.read_csv(os.path.join(args.logdir, ssrn.name, 'ckpt.csv'), sep=',', header=None)
    ckpt.columns = ['models', 'loss']
    ckpt = ckpt.sort_values(by='loss', ascending=True)
    state = torch.load(os.path.join(args.logdir, ssrn.name, ckpt.models.loc[0]))
    ssrn.load_state_dict(state['model'])

    print('All of models are loaded.')

    t2m.eval()
    ssrn.eval()
    
    if not os.path.exists(os.path.join(args.sampledir, 'A')):
        os.makedirs(os.path.join(args.sampledir, 'A'))
    return calculate_MSE(t2m=t2m, ssrn=ssrn, data_loader=data_loader, batch_size=args.mse_batch)



if __name__ == '__main__':

    gpu_id = int(sys.argv[1])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("Calculate MSE of train dataset...")
    train_mse, train_sample_number = main(mode="train")
    print("[TRAIN] MSE", train_mse, " for ", train_sample_number, " number")

    print("Calculate MSE of evaluation dataset...")
    eval_mse, eval_sample_number = main(mode="eval")
    print("[EVAL] MSE", eval_mse, " for ", eval_sample_number, " number")


    print("Calculate MSE of test dataset...")
    test_mse, test_sample_number = main(mode="test")
    print("[TEST] MSE", test_mse, " for ", test_sample_number, " number")

