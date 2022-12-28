# -*- coding: utf-8 -*-
import os
import json
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pypinyin import lazy_pinyin, Style

# cls sep unk pad
initial_ids = 'b、p、m、f、d、t、n、l、g、k、h、j、q、x、zh、ch、sh、r、z、c、s、y、w'.split('、')
final_ids = 'a o e i u v ai ei ui ao ou iu ie ve er an en in un vn ang eng ing ong'.split(' ')
initial_ids_dic = {initial_ids[i]: i+5 for i in range(len(initial_ids))}
final_ids_dic = {final_ids[i]: i+4 for i in range(len(final_ids))}
final_ids_dic['○'] = final_ids_dic['o']
# print(len(initial_ids_dic), len(final_ids_dic))

def is_Chinese(cp):
    if cp == '──':
        return False
    if ((cp >= '\u4E00' and cp <= '\u9FFF') or  #
            (cp >= '\u3400' and cp <= '\u4DBF') or  #
            (cp >= '\u20000' and cp <= '\u2A6DF') or  #
            (cp >= '\u2A700' and cp <= '\u2B73F') or  #
            (cp >= '\u2B740' and cp <= '\u2B81F') or  #
            (cp >= '\u2B820' and cp <= '\u2CEAF') or
            (cp >= '\uF900' and cp <= '\uFAFF') or  #
            (cp >= '\u2F800' and cp <= '\u2FA1F')):  #
        return True
    return False

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
        self.str2id = tk.get_vocab()
        self.id2str = {self.str2id[k]:k for k in self.str2id}
        self.data = open(path, 'r').read().split('\n')
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        # print(self.data[idx])
        # print(idx)
        # print(self.data[idx])
        src, trg, pos = self.data[idx].split('\t')
        
        pinyin = lazy_pinyin(src, style=Style.TONE3)
        if len(pinyin) > 126:
            pinyin = pinyin[:126]
        initial_ids = []
        final_ids = []
        tune_ids = []
       
        src0 = src
        n = len(src)
        #####  长度trunk一下
        src = self.tokenizer(src, padding="max_length", truncation=True, max_length=128)
        trg = self.tokenizer(trg, padding="max_length", truncation=True, max_length=128)['input_ids']
        d = self.d_tk(src0, padding="max_length", truncation=True, max_length=128)
        
        pos = []
        for i, j in zip(src['input_ids'], trg):
            if i == j:
                pos.append(0)
            else:
                pos.append(1)
            if i == 101: # cls
                initial_ids.append(1)
                final_ids.append(1)
                tune_ids.append(1)
            elif i == 102:# sep
                initial_ids.append(2)
                final_ids.append(2)
                tune_ids.append(2)
            elif i == 0: # pad
                initial_ids.append(0)
                final_ids.append(0)
                tune_ids.append(0)
            else:
                s = self.id2str[i]
                if is_Chinese(s):
                    pinyin = lazy_pinyin(s, style=Style.TONE3)[0]
                    if pinyin[-1].isdigit():
                        tune_ids.append(int(pinyin[-1])+3)
                        pinyin = pinyin[:-1]
                    else:
                        tune_ids.append(8)
                    if len(pinyin) == 1:
                        initial_ids.append(4)
                        final_ids.append(final_ids_dic[pinyin])
                    else:
                        if pinyin[:2] in initial_ids_dic:
                            initial_ids.append(initial_ids_dic[pinyin[:2]])
                            if pinyin[2:] in final_ids_dic:
                                final_ids.append(final_ids_dic[pinyin[2:]])
                            else:
                                final_ids.append(final_ids_dic[pinyin[3:]])
                        elif pinyin[:1] in initial_ids_dic:
                            initial_ids.append(initial_ids_dic[pinyin[:1]])
                            if pinyin[1:] in final_ids_dic:
                                final_ids.append(final_ids_dic[pinyin[1:]])
                            else:
                                if pinyin[2:] not in final_ids_dic:
                                    print('****')
                                    print(pinyin[2:])
                                    print(pinyin)
                                final_ids.append(final_ids_dic[pinyin[2:]])
                        else:
                            initial_ids.append(4)
                            if pinyin not in final_ids_dic or pinyin == '──':
                                print('****')
                                print(s)
                            final_ids.append(final_ids_dic[pinyin])
                else:
                    initial_ids.append(3)
                    final_ids.append(3)
                    tune_ids.append(3)
        out = {}
        
        out['labels'] = trg
        out['pos_labels'] = pos
        
        out['input_ids'] = src['input_ids']
        out['token_type_ids'] = src['token_type_ids']
        out['attention_mask'] = src['attention_mask']
        
        out['initial_ids'] = initial_ids
        out['final_ids'] = final_ids
        out['tune_ids'] = tune_ids
        
        out['d_input_ids'] = d['input_ids']
        out['d_token_type_ids'] = d['token_type_ids']
        out['d_attention_mask'] = d['attention_mask']
        
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
                 batch_size=32, num_workers=1, tk=None, train_13_14_15=None, d_tk=None):
    train_loader = None
    if train_path and os.path.exists(train_path):
        train_loader = DataLoader(ModelDataset(train_path, tk=tk, d_tk=d_tk),
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  )
    valid13 = None
    if valid13_path and os.path.exists(valid13_path):
        valid13 = DataLoader(ModelDataset(valid13_path, tk=tk, d_tk=d_tk),
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  )
    valid14 = None
    if valid14_path and os.path.exists(valid14_path):
        valid14 = DataLoader(ModelDataset(valid14_path, tk=tk, d_tk=d_tk),
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  )
    test15 = None
    if valid15_path and os.path.exists(valid15_path):
        test15 = DataLoader(ModelDataset(valid15_path, tk=tk, d_tk=d_tk),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 )
    if train_13_14_15:
        train_13_14_15_loader = DataLoader(ModelDataset(train_13_14_15, tk=tk, d_tk=d_tk),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 )
        return train_loader, valid13, valid14, test15, train_13_14_15_loader
    return train_loader, valid13, valid14, test15

if __name__ == '__main__':
    # tk = AutoTokenizer.from_pretrained('bert-base-chinese')
    # m = ModelDataset('/var/zgcCorrector/data/data/data/test14.txt', tk, tk)
    # for i in range(10):
    #     print(m.__getitem__(i)['input_ids'].shape)
    print(is_Chinese('──'))
