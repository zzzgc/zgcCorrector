import argparse
import functools
import os
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn.functional as F
from torch import nn
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_scheduler, AutoModel, AutoTokenizer
from transformers import BertForMaskedLM
from torch.utils.tensorboard import SummaryWriter

from dataset import dataset_demo
from modelling import DemoModel

from eval import stastics

global_step = 0
total_loss = 0
def train_loop(dataloader, model, optimizer, lr_scheduler, total_loss, args, valid_dataloader, if_pretrain=False):
    device = torch.device(args.device)
    # progress_bar = tqdm(range(len(dataloader)))
    # progress_bar.set_description(f'lr*10e3: {lr_scheduler.get_last_lr()[0]*10e3:>6f} loss: {0:>7f}')
    model.train()
    iter_items = tqdm(dataloader)
    for X in iter_items:
        x = {key:X[key].to(device) for key in X.keys()}
        loss = model(input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'], labels=x['tar_label'])
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        lr_scheduler.step()
        global global_step
        total_loss += loss.item()
        global_step += 1
        
        if if_pretrain:
            step = 1000
        else:
            step = 10
        if global_step % step == 1:
            print(total_loss/global_step, loss.item())
            test_loop(valid_dataloader, model, args)
        iter_items.set_postfix({'val_loss' : '{0:1.5f}'.format(loss)})


def test_loop(dataloader, model, args):
    device = torch.device(args.device)
    model.eval()
    src, tar, pre = [], [], []
    for X in tqdm(dataloader):
        x = {key:X[key].to(device) for key in X.keys()}
        pred0 = model(input_ids=x['input_ids'], token_type_ids=x['token_type_ids'], attention_mask=x['attention_mask'])
        src0 = x['input_ids'].tolist()
        pre0 = pred0.argmax(-1).tolist()
        labels=x['tar_label'].tolist()
        src += src0[:]
        tar += labels[:]
        pre += pre0[:]
    stastics(src, pre, tar)

def get_parser():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--batch_size", default=64, type=int,
                        help="batch size.")
    parser.add_argument("--step", default=1, type=int,
                        help="step to save")
    parser.add_argument("--device", default='cuda', type=str,
                        help="device")
    parser.add_argument("--epoch_num", default=100, type=int,
                        help="epoch numbers.")
    parser.add_argument("--pre_train_path", default='/var/zgcCorrector/data/src/pretrain_data_27w.txt', type=str)
    parser.add_argument("--fine_tune_path", default='/var/zgcCorrector/data/src/train_13_14_15.txt', type=str)
    parser.add_argument("--test_path", default='/var/zgcCorrector/data/src/test15.txt', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--learning_rate", default=0.00005, type=float,
                        help="learning rate.")
    parser.add_argument("--max_length", default=128, type=int,
                        help="sequence max length.")
    parser.add_argument("--model_name_or_path", default="hfl/chinese-macbert-base", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list:")
    parser.add_argument("--output_dir", default='output/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_ratio", default=0.9, type=float,
                        help="The train dataset ratio.")
    args = parser.parse_args()
    return args


def train(args):
    learning_rate = args.learning_rate
    epoch_num = args.epoch_num
    device = torch.device(args.device)
    model = DemoModel(args.model_name_or_path)
    tk = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model.to(device)
    pretrain_data = dataset_demo(args.pre_train_path, Tokenizer=tk)
    test_data = dataset_demo(args.test_path, Tokenizer=tk)
    finetune_data_data = dataset_demo(args.fine_tune_path, Tokenizer=tk)
    
    pretrain_dataloader = torch.utils.data.DataLoader(pretrain_data,
                                            batch_size=args.batch_size,
                                            num_workers=8,
                                            shuffle=True)
    finetune_dataloader = torch.utils.data.DataLoader(finetune_data_data,
                                            batch_size=args.batch_size,
                                            num_workers=8,
                                            shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                            batch_size=64,
                                            num_workers=8,
                                            shuffle=False)
   

    optimizer1 = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler1 = get_scheduler(
        "linear",
        optimizer=optimizer1,
        num_warmup_steps=(epoch_num)*len(pretrain_dataloader)*0.01,
        num_training_steps=(epoch_num)*len(pretrain_dataloader),
    )
    
    optimizer2 = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler2 = get_scheduler(
        "linear",
        optimizer=optimizer2,
        num_warmup_steps=(epoch_num)*len(finetune_dataloader)*0.01,
        num_training_steps=(epoch_num)*len(finetune_dataloader),
    )
    total_loss1, total_loss2 = 0, 0
    for t in range(epoch_num):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        train_loop(pretrain_dataloader, model, optimizer1, lr_scheduler1, total_loss1, args, test_dataloader, if_pretrain=True)
        train_loop(finetune_dataloader, model, optimizer2, lr_scheduler2, total_loss2, args, test_dataloader)
        test_loop(test_dataloader, model, args)

if __name__ == "__main__":
    args = get_parser()
    train(args)