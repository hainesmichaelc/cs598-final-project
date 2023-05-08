import pandas as pd
import math
import time

import pickle
import numpy as np
import sklearn.metrics as skm

import os
import importlib
import sys

import behrt_pretrain_model
importlib.reload(behrt_pretrain_model)
import behrt_pretrain_model
from behrt_pretrain_model import *


from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints")


class pretrain_behrt():
    def __init__(self, path="full"):
        if not os.path.exists(f"./checkpoints/{path}"):
            os.makedirs(f"./checkpoints/{path}")
        if not os.path.exists(f"./data/{path}/eval"):
            os.makedirs(f"./data/{path}/eval")
        def run_epoch(e, trainload, device):
            tr_loss = 0
            start = time.time()
            bert.train()
            for step, batch in enumerate(trainload):
                optim.zero_grad()
                batch = tuple(t for t in batch)
                input_ids, age_ids, gender_ids, ins_ids, eth_ids, posi_ids, segment_ids, attMask, masked_label = batch
                zeros = torch.zeros_like(input_ids)
                input_ids = torch.where(input_ids < zeros, zeros, input_ids)
                input_ids = input_ids.to(device)
                zeros = torch.zeros_like(age_ids)
                age_ids = torch.where(age_ids < zeros, zeros, age_ids)
                age_ids = age_ids.to(device)
                zeros = torch.zeros_like(gender_ids)
                gender_ids = torch.where(gender_ids < zeros, zeros, gender_ids)
                gender_ids = gender_ids.to(device)
                zeros = torch.zeros_like(eth_ids)
                eth_ids = torch.where(eth_ids < zeros, zeros, eth_ids)
                eth_ids = eth_ids.to(device)
                zeros = torch.zeros_like(ins_ids)
                ins_ids = torch.where(ins_ids < zeros, zeros, ins_ids)
                ins_ids = ins_ids.to(device)
                zeros = torch.zeros_like(posi_ids)
                posi_ids = torch.where(posi_ids < zeros, zeros, posi_ids)
                posi_ids = posi_ids.to(device)
                zeros = torch.zeros_like(segment_ids)
                segment_ids = torch.where(segment_ids < zeros, zeros, segment_ids)
                segment_ids = segment_ids.to(device)
                attMask=attMask.to(device)
                masked_label=masked_label.to(device)

                loss, pred, label = bert(input_ids, age_ids, gender_ids, ins_ids, eth_ids, posi_ids, segment_ids, attention_mask=attMask, masked_lm_labels=masked_label)
                loss.backward()
                tr_loss += loss.item()
                if step%500 == 0:
                    print(cal_acc(label, pred))
                optim.step()
                del loss, pred, label
            cost = time.time() - start
            return tr_loss, cost
        

        def train(trainload, valload, device):
            with open("log_pre_train.txt", 'w') as f:
                    f.write('')
            best_val = math.inf
            for e in range(train_params["epochs"]):
                print("Epoch n" + str(e))
                train_loss, train_time_cost = run_epoch(e, trainload, device)
                val_loss, val_time_cost,pred, label = eval(valload, device)
                train_loss = train_loss / math.ceil((train_params["train_data_len"]/train_params['batch_size']))
                val_loss = val_loss / math.ceil((train_params["val_data_len"]/train_params['batch_size']))
                print('TRAIN {}\t{} secs\n'.format(train_loss, train_time_cost))
                with open("log_pre_train.txt", 'a') as f:
                    f.write("Epoch n" + str(e) + '\n TRAIN {}\t{} secs\n'.format(train_loss, train_time_cost) + '\n\n\n')
                    f.write('EVAL {}\t{} secs\n'.format(val_loss, val_time_cost) + '\n\n\n')
                print('EVAL {}\t{} secs\n'.format(val_loss, val_time_cost))
                if val_loss < best_val:
                    print("** ** * Saving pre - trained model ** ** * ")
                    model_to_save = bert.module if hasattr(bert, 'module') else bert
                    save_model(model_to_save.state_dict(), f'./checkpoints/{path}/bert_pretrain')
                    best_val = val_loss
            return train_loss, val_loss

        def eval(_valload, device):
            tr_loss = 0
            start = time.time()
            bert.eval()
            for step, batch in enumerate(_valload):
                batch = tuple(t for t in batch)
                input_ids, age_ids, gender_ids, ins_ids, eth_ids, posi_ids, segment_ids, attMask, masked_label = batch
                zeros = torch.zeros_like(input_ids)
                input_ids = torch.where(input_ids < zeros, zeros, input_ids)
                input_ids = input_ids.to(device)
                zeros = torch.zeros_like(age_ids)
                age_ids = torch.where(age_ids < zeros, zeros, age_ids)
                age_ids = age_ids.to(device)
                zeros = torch.zeros_like(gender_ids)
                gender_ids = torch.where(gender_ids < zeros, zeros, gender_ids)
                gender_ids = gender_ids.to(device)
                zeros = torch.zeros_like(eth_ids)
                eth_ids = torch.where(eth_ids < zeros, zeros, eth_ids)
                eth_ids = eth_ids.to(device)
                zeros = torch.zeros_like(ins_ids)
                ins_ids = torch.where(ins_ids < zeros, zeros, ins_ids)
                ins_ids = ins_ids.to(device)
                zeros = torch.zeros_like(posi_ids)
                posi_ids = torch.where(posi_ids < zeros, zeros, posi_ids)
                posi_ids = posi_ids.to(device)
                zeros = torch.zeros_like(segment_ids)
                segment_ids = torch.where(segment_ids < zeros, zeros, segment_ids)
                segment_ids = segment_ids.to(device)
                attMask=attMask.to(device)
                masked_label=masked_label.to(device)

                loss, pred, label = bert(input_ids, age_ids, gender_ids, ins_ids, eth_ids, posi_ids, segment_ids, attention_mask=attMask, masked_lm_labels=masked_label)

                tr_loss += loss.item()
                del loss

            cost = time.time() - start
            return tr_loss, cost, pred, label

        def cal_acc(label, pred):
            logs = nn.LogSoftmax(dim=1)
            label=label.cpu().numpy()
            ind = np.where(label!=-1)[0]
            truepred = pred.detach().cpu().numpy()
            truepred = truepred[ind]
            truelabel = label[ind]
            truepred = logs(torch.tensor(truepred))
            outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
            precision = skm.precision_score(truelabel, outs, average='micro')
            return precision

        def save_model(_model_dict, file_name):
            torch.save(_model_dict, file_name)

        src = pd.read_csv(f"./data/{path}/tokens/tokenized_src.csv", header=None, index_col=0)
        age_data = pd.read_csv(f"./data/{path}/tokens/tokenized_age.csv", header=None, index_col=0)
        gender_data = pd.read_csv(f"./data/{path}/tokens/tokenized_gender.csv", header=None, index_col=0)
        ethnicity_data = pd.read_csv(f"./data/{path}/tokens/tokenized_ethni.csv", header=None, index_col=0)
        ins_data = pd.read_csv(f"./data/{path}/tokens/tokenized_ins.csv", header=None, index_col=0)
        mask_data = pd.read_csv(f"./data/{path}/tokens/tokenized_masks.csv", header=None, index_col=0)

        # Path to the pickle file = "./data/dict/srcDict"
        p_age = f"./data/{path}/dict/ageVocab"
        p_cond = f"./data/{path}/dict/condVocab"
        p_eth = f"./data/{path}/dict/ethVocab"
        p_ins = f"./data/{path}/dict/insVocab"
        p_med = f"./data/{path}/dict/medVocab"
        p_proc = f"./data/{path}/dict/procVocab"

        # Open the pickle file in read binary mode
        with open(p_age, "rb") as pickle_file:
            # Load the data from the pickle file
            age_vocab = pickle.load(pickle_file)

        with open(p_cond, "rb") as pickle_file:
            # Load the data from the pickle file
            cond_vocab = pickle.load(pickle_file)

        with open(p_eth, "rb") as pickle_file:
            # Load the data from the pickle file
            eth_vocab = pickle.load(pickle_file)

        with open(p_ins, "rb") as pickle_file:
            # Load the data from the pickle file
            ins_vocab = pickle.load(pickle_file)

        with open(p_med, "rb") as pickle_file:
            # Load the data from the pickle file
            med_vocab = pickle.load(pickle_file)

        with open(p_proc, "rb") as pickle_file:
            # Load the data from the pickle file
            proc_vocab = pickle.load(pickle_file)

        v_dict = {}
        for i, word in enumerate(age_vocab):
            v_dict[word] = word
        for i, word in enumerate(cond_vocab):
            v_dict[word] = i
        for i, word in enumerate(eth_vocab):
            v_dict[word] = i
        for i, word in enumerate(ins_vocab):
            v_dict[word] = i
        for i, word in enumerate(med_vocab):
            v_dict[word] = i
        for i, word in enumerate(proc_vocab):
            v_dict[word] = i
        vocab = v_dict
        vocab['MASK'] = int(max(vocab.values()))+1

        train_l = int(len(src)*0.70)
        val_l = int(len(src)*0.10)
        test_l = len(src) - val_l - train_l
        

        mask_data = []
        for index, row in src.iterrows():
            tokens, code, label = random_mask(row, vocab)
            mask_data.append(label)
        mask_data = pd.DataFrame(np.array(mask_data))

        mask_data.to_csv(f'./data/{path}/tokens/tokenized_masks.csv', header=None)

        global_params = {
            'max_seq_len': src.shape[1]
        }

        optim_param = {
            'lr_discr': 3e-5,
            'lr_gen': 3e-5
        }

        train_params = {
            'batch_size': 16,
            'use_cuda': True,
            'max_len_seq': global_params['max_seq_len'],
            'device': "cuda" if torch.cuda.is_available() else "cpu",
            'data_len' : len(src),
            'train_data_len' : train_l,
            'val_data_len' : val_l,
            'test_data_len' : test_l,
            'epochs' : 10,
            'action' : 'train',
        }

        model_config = {
            'vocab_size': int(src.max().max() + 1), # number of disease + symbols for word embedding
            'hidden_size': 828, # word embedding and seg embedding hidden size
            'seg_vocab_size': 2, # number of vocab for seg embedding
            'age_vocab_size': int(age_data.max().max() + 1), # number of vocab for age embedding
            'ins_vocab_size': int(ins_data.max().max()) + 1,
            'gender_vocab_size': 2,
            'eth_vocab_size': int(ethnicity_data.max().max()) + 1,
            'num_labels':1,
            'feature_dict':708,
            'max_position_embedding': train_params['max_len_seq'], # maximum number of tokens
            'hidden_dropout_prob': 0.2, # dropout rate
            'num_hidden_layers': 6, # number of multi-head attention layers required
            'num_attention_heads': 6, # number of attention heads
            'attention_probs_dropout_prob': 0.2, # multi-head attention dropout rate
            'intermediate_size': 256, # the size of the "intermediate" layer in the transformer encoder
            'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
            'initializer_range': 0.02, # parameter weight initializer range
            'number_output' : 1
        }

        train_code = src.values[:train_l]
        val_code = src.values[train_l:train_l + val_l]
        test_code = src.values[train_l + val_l:]

        train_age = age_data.values[:train_l]
        val_age = age_data.values[train_l:train_l + val_l]
        test_age = age_data.values[train_l + val_l:]

        train_gender = gender_data.values[:train_l]
        val_gender = gender_data.values[train_l:train_l + val_l]
        test_gender= gender_data.values[train_l + val_l:]

        train_ins = ins_data.values[:train_l]
        val_ins = ins_data.values[train_l:train_l + val_l]
        test_ins = ins_data.values[train_l + val_l:]

        train_eth = ethnicity_data.values[:train_l]
        val_eth = ethnicity_data.values[train_l:train_l + val_l]
        test_eth = ethnicity_data.values[train_l + val_l:]

        train_masks = mask_data.values[:train_l]
        val_masks = mask_data.values[train_l:train_l + val_l]
        test_masks = mask_data.values[train_l + val_l:]

        train_data = {"code":train_code, "age":train_age, "gender":train_gender, "ins":train_ins, "eth": train_eth, "masks": train_masks}
        val_data = {"code":val_code, "age":val_age, "gender":val_gender, "ins": val_ins, "eth": val_eth, "masks": val_masks}
        test_data = {"code":test_code,  "age":test_age, "gender":test_gender, "ins": test_ins, "eth": test_eth, "masks": test_masks}

        conf = BertConfig(model_config)
        bert = BertForMLM(conf)
        bert = bert.to(train_params['device'])

        bert_vars = [i for i in bert.parameters()]
        optim = torch.optim.Adam(bert_vars, lr=optim_param['lr_discr'])

        TrainDset = DataLoader(train_data, vocab, max_len=train_params['max_len_seq'], code='code')
        trainload = torch.utils.data.DataLoader(dataset=TrainDset, batch_size=train_params['batch_size'], shuffle=True)
        ValDset = DataLoader(val_data, vocab, max_len=train_params['max_len_seq'], code='code')
        valload = torch.utils.data.DataLoader(dataset=ValDset, batch_size=train_params['batch_size'], shuffle=True)
        train(trainload, valload, train_params['device'])