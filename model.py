from config import ConfigArgs as args
import torch
import torch.nn as nn
from network import TextEncoder, AudioEncoder, AudioDecoder, DotProductAttention
from torch.nn.utils import weight_norm as norm
import module as mm

import time as t

class Text2Mel(nn.Module):
    """
    Text2Mel
    Args:
        L: (N, Tx) text
        S: (N, Ty/r, n_mels) previous audio
    Returns:
        Y: (N, Ty/r, n_mels)
    """
    def __init__(self):
        super(Text2Mel, self).__init__()
        self.name = 'Text2Mel'
        self.embed = nn.Embedding(len(args.vocab), args.Ce, padding_idx=0)
        self.TextEnc = TextEncoder()
        self.AudioEnc = AudioEncoder()
        self.Attention = DotProductAttention()
        self.AudioDec = AudioDecoder()
        
        self.mean_textenc_time = 0
        self.mean_audioenc_time = 0
        self.mean_audiodec_time = 0
        self.mean_att_time = 0    


    def forward(self, L, S, is_print=[False, False]):

        L = self.embed(L).transpose(1,2) # -> (N, Cx, Tx) for conv1d
        S = S.transpose(1,2) # (N, n_mels, Ty/r) for conv1d

	# is_print[0]: test or not
        # is_print[1]: synthesize - last step or not


        if is_print[0] == True:
             text_enc_start_time = t.time()

        K, V = self.TextEnc(L) # (N, Cx, Tx) respectively

        if is_print[0] == True:
             text_enc_finish_time = t.time()
             self.mean_textenc_time += (text_enc_finish_time - text_enc_start_time)

        Q = self.AudioEnc(S) # -> (N, Cx, Ty/r)
 
        if is_print[0]== True:
             audio_enc_finish_time = t.time() 
             self.mean_audioenc_time += (audio_enc_finish_time - text_enc_finish_time)

        R, A = self.Attention(K, V, Q) # -> (N, Cx, Ty/r)
       
        if is_print[0] == True:
             att_finish_time = t.time()  
             self.mean_att_time += (att_finish_time - audio_enc_finish_time)

        R_ = torch.cat((R, Q), 1) # -> (N, Cx*2, Ty/r)
        Y = self.AudioDec(R_) # -> (N, n_mels, Ty/r)

        if is_print[0] == True:
             audio_dec_finish_time = t.time()
             self.mean_audiodec_time += (audio_dec_finish_time - att_finish_time)


        if is_print[1] == True:
             print("\n\t[Synthesis Result]")
             print("\t\tMean TextEncoder Time: ", self.mean_textenc_time / args.max_Ty)
             print("\t\tMean AudioEncoder Time: ", self.mean_audioenc_time / args.max_Ty)
             print("\t\tMean Attention Time: ", self.mean_att_time / args.max_Ty)
             print("\t\tMean AudioDecoder Time: ", self.mean_audiodec_time / args.max_Ty)


        return Y.transpose(1, 2), A # (N, Ty/r, n_mels)

class SSRN(nn.Module):
    """
    SSRN
    Args:
        Y: (N, Ty/r, n_mels)
    Returns:
        Z: (N, Ty, n_mags)
    """
    def __init__(self):
        super(SSRN, self).__init__()
        self.name = 'SSRN'
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
        
        Y = Y.transpose(1, 2) # -> (N, n_mels, Ty/r)
        Z = Y
        # -> (N, n_mags, Ty)
        for i in range(len(self.hc_blocks)):
            Z = self.hc_blocks[i](Z)
        Z = torch.sigmoid(Z)
        return Z.transpose(1, 2) # (N, Ty, n_mags)
