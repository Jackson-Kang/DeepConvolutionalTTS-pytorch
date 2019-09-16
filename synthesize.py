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
from data import TextDataset, synth_collate_fn, load_vocab
import utils
from scipy.io.wavfile import write
import time


def synthesize(t2m, ssrn, data_loader, batch_size=100):
    '''
    DCTTS Architecture
    Text --> Text2Mel --> SSRN --> Wav file
    '''

    text2mel_total_time = 0

    # Text2Mel
    idx2char = load_vocab()[-1]
    with torch.no_grad():
        print('='*10, ' Text2Mel ', '='*10)
        is_test = [True, False]
        total_mel_hats = torch.zeros([len(data_loader.dataset), args.max_Ty, args.n_mels]).to(DEVICE)
        mags = torch.zeros([len(data_loader.dataset), args.max_Ty*args.r, args.n_mags]).to(DEVICE)
        
        for step, (texts, mel, _) in enumerate(data_loader):
            texts = texts.to(DEVICE)
            prev_mel_hats = torch.zeros([len(texts), args.max_Ty, args.n_mels]).to(DEVICE)


            text2mel_start_time = time.time()         
            for t in tqdm(range(args.max_Ty-1), unit='B', ncols=70):
                if t == args.max_Ty - 2:
                    is_test[1] = True
                mel_hats, A, result_tuple = t2m(texts, prev_mel_hats, t, is_test) # mel: (N, Ty/r, n_mels)
                prev_mel_hats[:, t+1, :] = mel_hats[:, t, :]
		print(mel_hats.sum(), mel.sum())
            
            text2mel_finish_time = time.time()
            text2mel_total_time += (text2mel_finish_time - text2mel_start_time)

            total_mel_hats[step*batch_size:(step+1)*batch_size, :, :] = prev_mel_hats

            
            print('='*10, ' Alignment ', '='*10)
            alignments = A.cpu().detach().numpy()
            visual_texts = texts.cpu().detach().numpy()
            for idx in range(len(alignments)):
                text = [idx2char[ch] for ch in visual_texts[idx]]
                utils.plot_att(alignments[idx], text, args.global_step, path=os.path.join(args.sampledir, 'A'), name='{}.png'.format(idx))
            print('='*10, ' SSRN ', '='*10)
            # Mel --> Mag
            mags[step*batch_size:(step+1)*batch_size:, :, :] = \
                ssrn(total_mel_hats[step*batch_size:(step+1)*batch_size, :, :]) # mag: (N, Ty, n_mags)
            mags = mags.cpu().detach().numpy()
        print('='*10, ' Vocoder ', '='*10)
        for idx in trange(len(mags), unit='B', ncols=70):
            wav = utils.spectrogram2wav(mags[idx])
            write(os.path.join(args.sampledir, '{}.wav'.format(idx+1)), args.sr, wav)
 
    result = list(result_tuple)
    result.append(text2mel_total_time)

    return result

def main():

    testset = TextDataset(args.testset)
    test_loader = DataLoader(dataset=testset, batch_size=args.test_batch, drop_last=False,
                             shuffle=False, collate_fn=synth_collate_fn, pin_memory=True)

    t2m = Text2Mel().to(DEVICE)
    ssrn = SSRN().to(DEVICE)
    
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
    return synthesize(t2m=t2m, ssrn=ssrn, data_loader=test_loader, batch_size=args.test_batch)

if __name__ == '__main__':

    print("possible threads:", torch.get_num_threads())

    device_list = ['cpu', 'gpu']
    thread_num_list = [16]
    batch_size_list = [1]
    repeat_numb = 1

    gpu_id = int(sys.argv[1])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)

    f = open("./records/records.txt", "w")

    for device in device_list:

        print("device: ",  device)
        for batch_size in batch_size_list:
            print("\tbatch_size = ", batch_size)
            args.test_batch = batch_size

            for thread_num in thread_num_list:

                total_result=np.zeros(5, dtype=float)
            
                for i in range(repeat_numb):                 	
                    if device == "cpu":                        
                        DEVICE = torch.device("cpu")
                    else:
                        thread_num = 1
                        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    torch.set_num_threads(thread_num)
                    result_tuple = main()
                    total_result = np.add(total_result, result_tuple)

                print()
                total_result /= repeat_numb


                f.write("[ " + device.upper() + " ]"+ "\tbatch_size=" + str(batch_size) + "\tthread_numb=" + str(torch.get_num_threads())+"\n")
                f.write("\t\tTextEncoder: "+ str(total_result[0])+"\n")
                f.write("\t\tAudioEncoder: "+ str(total_result[1])+"\n")
                f.write("\t\tGuidedAttention: "+ str(total_result[2])+"\n")
                f.write("\t\tAudioDecoder: "+ str(total_result[3])+"\n")
                f.write("\tTotal Time: "+ str(total_result[4])+"\n\n")

                print()

                if device == "gpu":
                    break


            print()
        print()

    f.close()
