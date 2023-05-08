
import pandas as pd
import math
import time

import pickle
import numpy as np

import os
import importlib
import sys

from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints")

import behrt_no_d_model
importlib.reload(behrt_no_d_model)
import behrt_no_d_model
from behrt_no_d_model import *

class train_behrt():
    def __init__(self,src, target_data, path="full"):
        if not os.path.exists(f"./checkpoints/{path}"):
            os.makedirs(f"./checkpoints/{path}")
        if not os.path.exists(f"./data/{path}/eval"):
            os.makedirs(f"./data/{path}/eval")
        if not os.path.exists(f"./data/{path}/eval/behrt-no-d"):
            os.makedirs(f"./data/{path}/eval/behrt-no-d")

        def train(trainload, valload, device):
            best_val = math.inf
            for e in range(train_params["epochs"]):
                print("Epoch n" + str(e))
                train_loss, train_time_cost = run_epoch(e, trainload, device)
                val_loss, val_time_cost,pred, label = eval(valload, False, device)
                train_loss = train_loss / math.ceil((train_params["train_data_len"] / train_params['batch_size']))
                val_loss = val_loss / math.ceil((train_params["val_data_len"] / train_params['batch_size']))
                print('TRAIN {}\t{} secs\n'.format(train_loss, train_time_cost))
                print('EVAL {}\t{} secs\n'.format(val_loss, val_time_cost))
                if val_loss < best_val:
                    print("** ** * Saving fine - tuned model ** ** * ")
                    model_to_save = behrt.module if hasattr(behrt, 'module') else behrt
                    save_model(model_to_save.state_dict(), f'./checkpoints/{path}/behrt-no-d')
                    best_val = val_loss
                    TestDset = DataLoader(test_data, max_len=train_params['max_len_seq'], code='code')
                    testload = torch.utils.data.DataLoader(dataset=TestDset, batch_size=train_params['batch_size'], shuffle=False)
                    eval(testload, True, device)
            return train_loss, val_loss
        

        def run_epoch(e, trainload, device):
            tr_loss = 0
            start = time.time()
            behrt.train()
            for step, batch in enumerate(trainload):
                optim_behrt.zero_grad()
                batch = tuple(t for t in batch)
                input_ids, segment_ids, posi_ids, attMask, labels = batch
                zeros = torch.zeros_like(input_ids)
                input_ids = torch.where(input_ids < zeros, zeros, input_ids)
                input_ids = input_ids.to(device)
                zeros = torch.zeros_like(posi_ids)
                posi_ids = torch.where(posi_ids < zeros, zeros, posi_ids)
                posi_ids = posi_ids.to(device)
                zeros = torch.zeros_like(segment_ids)
                segment_ids = torch.where(segment_ids < zeros, zeros, segment_ids)
                segment_ids = segment_ids.to(device)
                attMask = attMask.to(device)
                labels = labels.to(device)

                logits = behrt(input_ids, segment_ids, posi_ids,
                               attention_mask=attMask)

                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                loss.backward()

                tr_loss += loss.item()
                if step%500 == 0:
                    print(loss.item())
                optim_behrt.step()
                del loss
            cost = time.time() - start
            return tr_loss, cost

        def eval(_valload, saving, device):
            tr_loss = 0
            start = time.time()
            behrt.eval()
            if saving:
                with open(f"./data/{path}/eval/behrt-no-d/preds.csv", 'w') as f:
                    f.write('')
                with open(f"./data/{path}/eval/behrt-no-d/labels.csv", 'w') as f:
                    f.write('')

            for step, batch in enumerate(_valload):
                batch = tuple(t for t in batch)
                input_ids, segment_ids, posi_ids, attMask, labels = batch
                zeros = torch.zeros_like(input_ids)
                input_ids = torch.where(input_ids < zeros, zeros, input_ids)
                input_ids = input_ids.to(device)
                zeros = torch.zeros_like(posi_ids)
                posi_ids = torch.where(posi_ids < zeros, zeros, posi_ids)
                posi_ids = posi_ids.to(device)
                zeros = torch.zeros_like(segment_ids)
                segment_ids = torch.where(segment_ids < zeros, zeros, segment_ids)
                segment_ids = segment_ids.to(device)
                attMask = attMask.to(device)
                labels = labels.to(device)

                logits = behrt(input_ids, segment_ids, posi_ids,
                               attention_mask=attMask)

                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

                if saving:
                    with open(f"./data/{path}/eval/behrt-no-d/preds.csv", 'a') as f:
                        pd.DataFrame(logits.detach().cpu().numpy()).to_csv(f, header=False)
                    with open(f"./data/{path}/eval/behrt-no-d/labels.csv", 'a') as f:
                        pd.DataFrame(labels.detach().cpu().numpy()).to_csv(f, header=False)

                tr_loss += loss.item()
                del loss

            print("TOTAL LOSS", tr_loss)

            cost = time.time() - start
            return tr_loss, cost, logits, labels

        def save_model(_model_dict, file_name):
            torch.save(_model_dict, file_name)


        train_l = int(len(src)*0.70)
        val_l = int(len(src)*0.1)
        test_l = len(src) - val_l - train_l
        number_output = target_data.shape[1]

        file_config = {
            'model_path': './saved_models/', # where to save model
            'model_name': 'CVDTransformer', # model name
            'file_name': 'log.txt',  # log path
        }
        #create_folder(file_config['model_path'])

        global_params = {
            'max_seq_len': src.shape[1],
            'month': 1,
            'min_visit': 3,
            'gradient_accumulation_steps': 1
        }

        optim_param = {
            'lr': 3e-5,
            'warmup_proportion': 0.1,
            'weight_decay': 0.01
        }

        train_params = {
            'batch_size': 16,
            'use_cuda': True,
            'max_len_seq': global_params['max_seq_len'],
            'device': "cuda:0" if torch.cuda.is_available() else "cpu",
            'data_len' : len(target_data),
            'train_data_len' : train_l,
            'val_data_len' : val_l,
            'test_data_len' : test_l,
            'epochs' : 10, #change back to 10
            'action' : 'train'
        }

        model_config = {
            'vocab_size': int(src.max().max() + 1), # number of disease + symbols for word embedding
            'hidden_size': 828, # word embedding and seg embedding hidden size
            'seg_vocab_size': 2, # number of vocab for seg embedding
            'max_position_embedding': train_params['max_len_seq'], # maximum number of tokens
            'hidden_dropout_prob': 0.2, # dropout rate
            'num_hidden_layers': 6, # number of multi-head attention layers required
            'num_attention_heads': 6, # number of attention heads
            'attention_probs_dropout_prob': 0.2, # multi-head attention dropout rate
            'intermediate_size': 256, # the size of the "intermediate" layer in the transformer encoder
            'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
            'initializer_range': 0.02, # parameter weight initializer range
            'number_output' : number_output
        }

        print("Training with config: ", model_config)

        train_code = src.values[:train_l]
        val_code = src.values[train_l:train_l + val_l]
        test_code = src.values[train_l + val_l:]

        train_labels = target_data.values[:train_l]
        val_labels = target_data.values[train_l:train_l + val_l]
        test_labels = target_data.values[train_l + val_l:]

        train_data = {"code":train_code, "labels":train_labels}
        val_data = {"code":val_code, "labels":val_labels}
        test_data = {"code":test_code, "labels":test_labels}

        conf = BertConfig(model_config)
        behrt = BertForEHRPrediction(conf, model_config['number_output'])

        behrt = behrt.to(train_params['device'])

        #models parameters
        transformer_vars = [i for i in behrt.parameters()]

        #optimizer
        optim_behrt = torch.optim.Adam(transformer_vars, lr=3e-5)

        TrainDset = DataLoader(train_data, max_len=train_params['max_len_seq'], code='code')
        trainload = torch.utils.data.DataLoader(dataset=TrainDset, batch_size=train_params['batch_size'], shuffle=True)
        ValDset = DataLoader(val_data, max_len=train_params['max_len_seq'], code='code')
        valload = torch.utils.data.DataLoader(dataset=ValDset, batch_size=train_params['batch_size'], shuffle=True)

        train_loss, val_loss = train(trainload, valload, train_params['device'])