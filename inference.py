
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
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--speaker", type=int, default=0,
)
parser.add_argument("-m", "--model", type=str, default="./pretrained_vctk.pth")
parser.add_argument("-c", "--config", type=str, default="./configs/vctk_base.json",)
parser.add_argument("-d", "--device", type=str, default="cuda",)
parser.add_argument("-hml", "--hpu-max-length", type=int, default=2048)

args, _ = parser.parse_known_args()

if args.device == "hpu":
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.gpu_migration

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
    **hps.model).to(device=args.device)
_ = net_g.eval()

_ = utils.load_checkpoint(args.model, net_g, None)

config_name = args.config.split("/")[-1].split(".")[-2]
model_name = args.model.split("/")[-1].split(".")[-2]

# Warmup to max_hpu_len for HPU
max_hpu_len = args.hpu_max_length

with torch.no_grad():
    if args.device == "hpu":
        print("Warmup HPU...")
        texts = ["Build your chatbot within minutes on your favorite device.", "offer SOTA compression techniques for LLMs.", " run LLMs efficiently on Intel Platforms."]
        for i in range(len(texts)):
            S_T = time.time()
            stn_tst = get_text(texts[i], hps)
            concrete_lengths = stn_tst.size(0)

            stn_tst = torch.nn.functional.pad(stn_tst, (0, max_hpu_len-stn_tst.size(0)), value=0)
            x_tst = stn_tst.to(device=args.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([concrete_lengths]).to(device=args.device)
            sid = torch.LongTensor([args.speaker]).to(device=args.device)
            result = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1, max_len=max_hpu_len)
            print(f"Warmup time: {time.time()-S_T}")

# Inference begin
while True:

    try:
        text = input("Write a sentence to let VITS speak:\n")
        S_T = time.time()
        stn_tst = get_text(text, hps)
        with torch.no_grad():

            if args.device == "hpu":
                concrete_lengths = stn_tst.size(0)
                print(f"concrete x lengths: {concrete_lengths}")
                print(f"pad x lengths: {max_hpu_len}")
                stn_tst = torch.nn.functional.pad(stn_tst, (0, max_hpu_len-stn_tst.size(0)), value=0)
                x_tst = stn_tst.to(device=args.device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([concrete_lengths]).to(device=args.device)
                sid = torch.LongTensor([args.speaker]).to(device=args.device)
                result = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1, max_len=max_hpu_len)
                # filter the concrete part
                audio = result[0][0,0].data.cpu().float().numpy()
                print(f"concrete y lengths: {len(result[2].nonzero())}")
                print(f"pad y lengths: {len(result[2][0][0])}")
                audio = audio[:int(audio.shape[0]*len(result[2].nonzero())/len(result[2][0][0]))]
                print(f"Inference time: {time.time()-S_T}, audio shape: {audio.shape}")
            else:
                x_tst = stn_tst.to(device=args.device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device=args.device)
                sid = torch.LongTensor([args.speaker]).to(device=args.device)

                audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1,
                                    max_len=None)[0][0,0].data.cpu().float().numpy()
            print(f"Write to output_{config_name}_{model_name}_{int(sid)}.wav")
            write(f"output_{config_name}_{model_name}_{int(sid)}.wav", rate=hps.data.sampling_rate, data=audio)
            print(f"Inference time: {time.time()-S_T}")
    except Exception as e:
        print(f"Catch exception: {e}")
        print("Restarting\n")
