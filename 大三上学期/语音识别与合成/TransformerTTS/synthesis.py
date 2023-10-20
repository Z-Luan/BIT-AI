import torch as t
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse

# 加载保存的模型参数
def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('./checkpoint/checkpoint_%s_%d.pth.tar'% (model_name, step))   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value
    return new_state_dict

def synthesis(text, args):
    m = Model()
    m_post = ModelPostNet()
    # Transformer
    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))
    # PosNet
    m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))
    # 字符序列输入
    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    # 增加一个 batch 维度
    text = t.LongTensor(text).unsqueeze(0) 
    text = text.cuda()
    mel_input = t.zeros([1, 1, 80]).cuda()
    pos_text = t.arange(1, text.size(1) + 1).unsqueeze(0) # 用于构建mask
    pos_text = pos_text.cuda()
    m = m.cuda()
    m_post = m_post.cuda()
    m.train(False)
    m_post.train(False)
    # max_len 设置最大的预测长度
    pbar = tqdm(range(args.max_len))
    with t.no_grad():
        for i in pbar:
            pos_mel = t.arange(1, mel_input.size(1) + 1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(text, mel_input, pos_text, pos_mel)
            # 将每次预测生成的 mel 与之前的 mel 进行 cat 作为一下次计算的输入
            mel_input = t.cat([mel_input, mel_pred[:,-1:,:]], dim=1) 
        # 使用最后输出的经过 postconvnet 处理的 mel 谱图生成 mag 谱图
        mag_pred = m_post.forward(postnet_pred) 
    # 基于 mag 谱图生成音频
    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    write(hp.sample_path + "/test.wav", hp.sr, wav)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=80000)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint', default=80000)
    parser.add_argument('--max_len', type=int, help='Global step to restore checkpoint', default=200)
    args = parser.parse_args()
    synthesis("happy new year.",args)
