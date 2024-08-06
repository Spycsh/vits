
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--speaker", type=int, default=0,
)
parser.add_argument("-m", "--model", type=str, default="./pretrained_vctk.pth")
parser.add_argument("-c", "--config", type=str, default="./configs/vctk_base.json",)
parser.add_argument("-t", "--text", type=str, default="I hope all of you are fine!",)


args, _ = parser.parse_known_args()

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

##### ljspeech single speaker

# hps = utils.get_hparams_from_file("./configs/ljs_base.json")

# net_g = SynthesizerTrn(
#     len(symbols),
#     hps.data.filter_length // 2 + 1,
#     hps.train.segment_size // hps.data.hop_length,
#     **hps.model).cuda()
# _ = net_g.eval()

# _ = utils.load_checkpoint("./pretrained_ljs.pth", net_g, None)

# stn_tst = get_text("After LLMs NIC OPEA BIC Mac Intel DDP Java Python C++ SOTA installation, set the system environment variables and you are done.", hps)
# with torch.no_grad():
#     x_tst = stn_tst.cuda().unsqueeze(0)
#     x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
#     audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

# from scipy.io.wavfile import write

# write("output_ljspeech.wav", rate=hps.data.sampling_rate, data=audio)

############ vctk 110 speakers

hps = utils.get_hparams_from_file(args.config)

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint(args.model, net_g, None)

stn_tst = get_text(args.text, hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([args.speaker]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

config_name = args.config.split("/")[-1].split(".")[-2]
model_name = args.model.split("/")[-1].split(".")[-2]
write(f"output_{config_name}_{model_name}_{int(sid)}.wav", rate=hps.data.sampling_rate, data=audio)