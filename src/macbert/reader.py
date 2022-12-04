# -*- coding: utf-8 -*-
import os
import json
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from transformers import AutoTokenizer


initial_ids = 'b、p、m、f、d、t、n、l、g、k、h、j、q、x、zh、ch、sh、r、z、c、s、y、w'.split('、')
final_ids = 'a o e i u v ai ei ui ao ou iu ie ve er an en in un vn ang eng ing ong'.split(' ')
initial_ids = {initial_ids[i]: i+1 for i in range(initial_ids)}
final_ids = {final_ids[i]: i+1 for i in range(final_ids)}
class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        ori_texts, cor_texts, wrong_idss = zip(*data)
        encoded_texts = [self.tokenizer.tokenize(t) for t in ori_texts]
        max_len = max([len(t) for t in encoded_texts]) + 2
        det_labels = torch.zeros(len(ori_texts), max_len).long()
        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            for idx in wrong_ids:
                margins = []
                for word in encoded_text[:idx]:
                    if word == '[UNK]':
                        break
                    if word.startswith('##'):
                        margins.append(len(word) - 3)
                    else:
                        margins.append(len(word) - 1)
                margin = sum(margins)
                move = 0
                while (abs(move) < margin) or (idx + move >= len(encoded_text)) \
                        or encoded_text[idx + move].startswith('##'):
                    move -= 1
                det_labels[i, idx + move + 1] = 1
        return ori_texts, cor_texts, det_labels

class ModelDataset(Dataset):
    def __init__(self, path, tk, d_tk):
        self.tokenizer = tk
        # if path == '/var/zgcCorrector/data/data/sighan_27w.txt':
        #     p2 = "/var/zgcCorrector/data/data/13_14_15.txt"
        #     self.data = open(path, 'r').read().split('\n') + open(p2, 'r').read().strip().split('\n')+ open(p2, 'r').read().strip().split('\n')+ open(p2, 'r').read().strip().split('\n')+ open(p2, 'r').read().strip().split('\n')
        # else:
        self.d_tk = d_tk
        self.data = open(path, 'r').read().split('\n')
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        # print(self.data[idx])
        # print(idx)
        # print(self.data[idx])
        src, trg, pos = self.data[idx].split('\t')
        n = len(src)
        #####  长度trunk一下
        src = self.tokenizer(src, padding="max_length", truncation=True, max_length=128)
        trg = self.tokenizer(trg, padding="max_length", truncation=True, max_length=128)['input_ids']
        d = self.tokenizer(trg, padding="max_length", truncation=True, max_length=128)['input_ids']
        
        if len(pos)>126:
            pos = pos[:126]
            n = 126
        pos = [0] + [int(i) for i in pos] + [0] + [2] * (126 - len(pos))
        out = {}
        
        out['labels'] = trg
        out['pos_labels'] = pos
        out['input_ids'] = src['input_ids']
        out['token_type_ids'] = src['token_type_ids']
        out['attention_mask'] = src['attention_mask']
        out['len'] = n
        
        return {key: torch.tensor(value).to('cuda') for key, value in out.items()}

class CscDataset(Dataset):
    def __init__(self, file_path):
        self.data = json.load(open(file_path, 'r', encoding='utf-8'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['original_text'], self.data[index]['correct_text'], self.data[index]['wrong_ids']


def make_loaders(train_path='', valid13_path='', valid14_path='', valid15_path='',
                 batch_size=32, num_workers=0, tk=None, train_13_14_15=None):
    train_loader = None
    if train_path and os.path.exists(train_path):
        train_loader = DataLoader(ModelDataset(train_path, tk=tk),
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  )
    valid13 = None
    if valid13_path and os.path.exists(valid13_path):
        valid13 = DataLoader(ModelDataset(valid13_path, tk=tk),
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  )
    valid14 = None
    if valid14_path and os.path.exists(valid14_path):
        valid14 = DataLoader(ModelDataset(valid14_path, tk=tk),
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  )
    test15 = None
    if valid15_path and os.path.exists(valid15_path):
        test15 = DataLoader(ModelDataset(valid15_path, tk=tk),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 )
    if train_13_14_15:
        train_13_14_15_loader = DataLoader(ModelDataset(train_13_14_15, tk=tk),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 )
        return train_loader, valid13, valid14, test15, train_13_14_15_loader
    return train_loader, valid13, valid14, test15

if __name__ == '__main__':
    tk = AutoTokenizer.from_pretrained('bert-base-chinese')
    m = ModelDataset('/home/ygq/zgc/zgcCorrector-master/data/test15.txt', tk)
    for i in range(10):
        print(m.__getitem__(i)['input_ids'].shape)
