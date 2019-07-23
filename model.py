from config import ConfigArgs as args
import torch
import torch.nn as nn
from network import TextEncoder, AudioEncoder, AudioDecoder, DotProductAttention
from torch.nn.utils import weight_norm as norm
import module as mm
import time


class Text2Mel(nn.Module):
    """
    Text2Mel
    Args:
        L: (N, Tx) text
        S: (N, Ty/r, n_mels) previous audio
    Returns:
        Y: (N, Ty/r, n_mels)
    """
    def __init__(self, is_test):
        super(Text2Mel, self).__init__()
        self.name = 'Text2Mel'
        self.embed = nn.Embedding(len(args.vocab), args.Ce, padding_idx=0)
        self.TextEnc = TextEncoder()
        self.AudioEnc = AudioEncoder()
        self.Attention = DotProductAttention()
        self.AudioDec = AudioDecoder()
    
        self.is_test = is_test

        self.text_enc_mean = 0
        self.aud_enc_mean = 0
        self.att_mean = 0
        self.aud_dec_mean = 0
        self.count = 0


    def forward(self, L, S, is_print = False):
        L = self.embed(L).transpose(1,2) # -> (N, Cx, Tx) for conv1d
        S = S.transpose(1,2) # (N, n_mels, Ty/r) for conv1d

        text_enc_start = time.time()
        K, V = self.TextEnc(L) # (N, Cx, Tx) respectively
        text_enc_finish = time.time()

        if self.is_test:
            self.count+=1
            self.text_enc_mean += (text_enc_finish - text_enc_start)

            if is_print:
                print("\n\twhole network was iterated by ", self.count)
                print("\t\t[Text Encoder] Mean Execution time: ", self.text_enc_mean/self.count)

        Q = self.AudioEnc(S) # -> (N, Cx, Ty/r)
        audio_enc_finish = time.time()
	
	
        if self.is_test:
            self.aud_enc_mean += (audio_enc_finish - text_enc_finish)

            if is_print:
                print("\n\t\t[Audio Encoder] Mean Execution time: ", self.aud_enc_mean/self.count)


        R, A = self.Attention(K, V, Q) # -> (N, Cx, Ty/r)
        attention_finish = time.time()
	
        if self.is_test:
            self.att_mean += (attention_finish - audio_enc_finish)

            if is_print:
                print("\n\t\t[Attention] Mean Execution time: ", self.att_mean/self.count)

        R_ = torch.cat((R, Q), 1) # -> (N, Cx*2, Ty/r)

        Y = self.AudioDec(R_) # -> (N, n_mels, Ty/r)
        audio_dec_finish = time.time()
	
        if self.is_test:
            self.aud_dec_mean += (audio_dec_finish - attention_finish)            
            if is_print:
                print("\n\t\t[Audio Decoder] Execution time: ", self.aud_dec_mean / self.count)

        return Y.transpose(1, 2), A # (N, Ty/r, n_mels)

class SSRN(nn.Module):
    """
    SSRN
    Args:
        Y: (N, Ty/r, n_mels)
    Returns:
        Z: (N, Ty, n_mags)
    """
    def __init__(self, is_test):
        super(SSRN, self).__init__()
        self.name = 'SSRN'
        self.is_test = is_test
        self.ssrn_mean = 0

        # (N, n_mels, Ty/r) -> (N, Cs, Ty/r)
        self.hc_blocks = nn.ModuleList([norm(mm.Conv1d(args.n_mels, args.Cs, 1, activation_fn=torch.relu))])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cs, args.Cs, 3, dilation=3**i))
                               for i in range(2)])
        # (N, Cs, Ty/r*2) -> (N, Cs, Ty/r*2)
        self.hc_blocks.extend([norm(mm.ConvTranspose1d(args.Cs, args.Cs, 4, stride=2, padding=1))])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cs, args.Cs, 3, dilation=3**i))
                               for i in range(2)])
        # (N, Cs, Ty/r*2) -> (N, Cs, Ty/r*4==Ty)
        self.hc_blocks.extend([norm(mm.ConvTranspose1d(args.Cs, args.Cs, 4, stride=2, padding=1))])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cs, args.Cs, 3, dilation=3**i))
                               for i in range(2)])
        # (N, Cs, Ty) -> (N, Cs*2, Ty)
        self.hc_blocks.extend([norm(mm.Conv1d(args.Cs, args.Cs*2, 1))])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cs*2, args.Cs*2, 3, dilation=1))
                               for i in range(2)])
        # (N, Cs*2, Ty) -> (N, n_mags, Ty)
        self.hc_blocks.extend([norm(mm.Conv1d(args.Cs*2, args.n_mags, 1))])
        self.hc_blocks.extend([norm(mm.Conv1d(args.n_mags, args.n_mags, 1, activation_fn=torch.relu))
                               for i in range(2)])
        self.hc_blocks.extend([norm(mm.Conv1d(args.n_mags, args.n_mags, 1))])

    def forward(self, Y):
        ssrn_start = time.time()

        Y = Y.transpose(1, 2) # -> (N, n_mels, Ty/r)
        Z = Y
        # -> (N, n_mags, Ty)
        for i in range(len(self.hc_blocks)):
            Z = self.hc_blocks[i](Z)
            Z = torch.sigmoid(Z)

        if self.is_test:
            print("\n\t\t[SSRN] Execution time: ", (time.time()-ssrn_start)/i+1)


        return Z.transpose(1, 2) # (N, Ty, n_mags)
