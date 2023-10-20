from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm

# 用于训练 Transformer 网络(text --> mel)
# 动态调整学习率
def adjust_learning_rate(optimizer, step_num, warmup_step = 4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    dataset = get_dataset() # 获得训练数据
    global_step = 0
    m = nn.DataParallel(Model().cuda()) # 初始化模型 如果有多个 GPU，在多个 GPU 上并行训练
    m.train()
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)
    writer = SummaryWriter() # 初始化 tensorboard 中的对象
    for epoch in range(hp.epochs):
        # collate_fn 用于 padding 保证可以批处理训练
        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=1)
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d"%epoch)
            # 当 global_step 小于 400000 之前，每个 batch 训练时都进行 lr 调整
            global_step += 1 
            if global_step < 400000: 
                adjust_learning_rate(optimizer, global_step)
            character, mel, mel_input, pos_text, pos_mel, _ = data
            # 将数据都传入GPU中
            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()
            # 前向传播
            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(character, mel_input, pos_text, pos_mel)
            mel_loss = nn.L1Loss()(mel_pred, mel) # 未经 postconvnet 处理的 mel 的损失
            post_mel_loss = nn.L1Loss()(postnet_pred, mel) # 经过 postconvnet 处理的 mel 的损失
            loss = mel_loss + post_mel_loss
            writer.add_scalars('training_loss',{'mel_loss':mel_loss, 'post_mel_loss':post_mel_loss,}, global_step) # 记录训练过程中的损失
            writer.add_scalars('alphas',{'encoder_alpha':m.module.encoder.alpha.data, 'decoder_alpha':m.module.decoder.alpha.data,}, global_step) # 记录训练时位置编码中参数 alpha
            if global_step % hp.image_step == 1: # 每训练 500 个 batch
                for i, prob in enumerate(attn_probs): # 将 decoder 中的交叉注意力保存为图像
                    num_h = prob.size(0)
                    for j in range(4):
                        x = vutils.make_grid(prob[j*4] * 255)
                        writer.add_image('Attention_%d_0'%global_step, x, i*4+j)
                for i, prob in enumerate(attns_enc): # 将 encoder 中的自注意力保存为图像
                    num_h = prob.size(0)
                    for j in range(4):
                        x = vutils.make_grid(prob[j*4] * 255)
                        writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j)
                for i, prob in enumerate(attns_dec): # 将 decoder 中的自注意力保存为图像
                    num_h = prob.size(0)
                    for j in range(4):
                        x = vutils.make_grid(prob[j*4] * 255)
                        writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j)
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.) # 梯度裁剪
            # 参数更新
            optimizer.step()
            if global_step % hp.save_step == 0: # 每 2000 个 batch 进行一次权重保存
                t.save({'model':m.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_transformer_%d.pth.tar' % global_step))

if __name__ == '__main__':
    main()