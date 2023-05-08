import pandas as pd
import math
import time

import pickle
import numpy as np
import sklearn.metrics as skm
from sklearn.model_selection import KFold

import os
import importlib
import sys

import behrt_finetune_model
importlib.reload(behrt_finetune_model)
from behrt_finetune_model import  *

from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("./checkpoints"):
    os.makedirs("./checkpoints")


class finetune_behrt():
    def __init__(self, path="full"):
        if not os.path.exists(f"./checkpoints/{path}"):
            os.makedirs(f"./checkpoints/{path}")
        if not os.path.exists(f"./data/{path}/eval"):
            os.makedirs(f"./data/{path}/eval")
        if not os.path.exists(f"./data/{path}/eval/custom"):
            os.makedirs(f"./data/{path}/eval/custom")

        src = pd.read_csv(f"./data/{path}/tokens/tokenized_src.csv", header=None, index_col=0)
        age_data = pd.read_csv(f"./data/{path}/tokens/tokenized_age.csv", header=None, index_col=0)
        gender_data = pd.read_csv(f"./data/{path}/tokens/tokenized_gender.csv", header=None, index_col=0)
        ethnicity_data = pd.read_csv(f"./data/{path}/tokens/tokenized_ethni.csv", header=None, index_col=0)
        ins_data = pd.read_csv(f"./data/{path}/tokens/tokenized_ins.csv", header=None, index_col=0)
        mask_data = pd.read_csv(f"./data/{path}/tokens/tokenized_masks.csv", header=None, index_col=0)
        target_data = pd.read_csv(f"./data/{path}/tokens/tokenized_labels.csv", header=None, index_col=0)

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
        number_output = target_data.shape[1]
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
            'data_len' : len(target_data),
            'train_data_len' : train_l,
            'val_data_len' : val_l,
            'test_data_len' : test_l,
            'epochs' : 10,
            'action' : 'train',
            'alpha_unsup':0.35
        }

        model_config = {
            'vocab_size': int(src.max().max() + 1), # number of disease + symbols for word embedding
            'hidden_size': 828, # word embedding and seg embedding hidden size
            'seg_vocab_size': 2, # number of vocab for seg embedding
            'age_vocab_size': max(age_vocab) + 1, # number of vocab for age embedding
            'ins_vocab_size': int(ins_data.max().max()) + 1,
            'gender_vocab_size': 2,
            'ethnicity_vocab_size': int(len(eth_vocab) + 1),
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

        kf = KFold(n_splits=5, random_state=None)

        k = 5
        i = 1
        few_shots = 1

        for train_index, test_index in kf.split(src):

            amount_few_shots = round(train_l * few_shots)
            val_index = train_index[train_l:]
            train_index= train_index[:train_l]
            train_index = train_index[np.random.choice(len(train_index), size=amount_few_shots, replace=False)]

            train_code = src.values[train_index]
            val_code = src.values[val_index]
            test_code = src.values[test_index]

            train_age = age_data.values[train_index]
            val_age = age_data.values[val_index]
            test_age = age_data.values[test_index]

            train_gender = gender_data.values[train_index]
            val_gender = gender_data.values[val_index]
            test_gender= gender_data.values[test_index]

            train_ethnicity = ethnicity_data.values[train_index]
            val_ethnicity = ethnicity_data.values[val_index]
            test_ethnicity = ethnicity_data.values[test_index]

            train_ins = ins_data.values[train_index]
            val_ins = ins_data.values[val_index]
            test_ins = ins_data.values[test_index]

            train_labels = target_data.values[train_index]
            val_labels = target_data.values[val_index]
            test_labels = target_data.values[test_index]

            train_masks = mask_data.values[train_index]
            val_masks = mask_data.values[val_index]
            test_masks= mask_data.values[test_index]

            if i == k:
                train_data = {"code":train_code, "age":train_age, "gender":train_gender, "ethnicity":train_ethnicity, "ins": train_ins, "labels":train_labels, "masks":train_masks}
                val_data = {"code":val_code, "age":val_age, "gender":val_gender, "ethnicity":val_ethnicity, "ins":val_ins, "labels":val_labels, "masks":val_masks}
                test_data = {"code":test_code,  "age":test_age, "gender":test_gender, "ethnicity":test_ethnicity, "ins":test_ins, "labels":test_labels, "masks": test_masks}
                break
            i+=1
        
        noise_size=100
        hidden_size=828
        hidden_levels_d=[828, 828]
        hidden_levels_g=[828, 828]
        out_dropout_rate = 0.2

        conf = BertConfig(model_config)
        bert = BertForEHR(conf)
        generator = Generator(noise_size=noise_size, output_size=hidden_size, hidden_sizes=hidden_levels_g, dropout_rate=out_dropout_rate)
        discriminator = Discr(input_size=hidden_size, hidden_sizes=hidden_levels_d, num_labels=number_output, dropout_rate=out_dropout_rate)

        discriminator = discriminator.to(train_params['device'])
        generator = generator.to(train_params['device'])
        bert = bert.to(train_params['device'])

        #models parameters
        transformer_vars = [i for i in bert.parameters()]
        d_vars = transformer_vars + [v for v in discriminator.parameters()]
        g_vars = [v for v in generator.parameters()]

        #optimizer
        optim_disc_behrt = torch.optim.Adam(d_vars, lr=optim_param['lr_discr'])
        optim_gen = torch.optim.AdamW(g_vars, lr=optim_param['lr_gen'])


        bce_loss = nn.BCELoss()
        bce_logits_loss = nn.BCEWithLogitsLoss(reduction='none')


        def run_epoch(e, trainload, device):
            tr_loss = 0
            start = time.time()
            bert.train()
            discriminator.train()
            generator.train()
            for step, batch in enumerate(trainload):
                batch = tuple(t for t in batch)
                input_ids, age_ids, gender_ids, ethnicity_ids, ins_ids, posi_ids, segment_ids, attMask, labels, masks = batch
                
                zeros = torch.zeros_like(input_ids)
                input_ids = torch.where(input_ids < zeros, zeros, input_ids)
                input_ids = input_ids.to(device)
                zeros = torch.zeros_like(age_ids)
                age_ids = torch.where(age_ids < zeros, zeros, age_ids)
                age_ids = age_ids.to(device)
                zeros = torch.zeros_like(gender_ids)
                gender_ids = torch.where(gender_ids < zeros, zeros, gender_ids)
                gender_ids = gender_ids.to(device)
                zeros = torch.zeros_like(ethnicity_ids)
                ethnicity_ids = torch.where(ethnicity_ids < zeros, zeros, ethnicity_ids)
                ethnicity_ids = ethnicity_ids.to(device)
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
                labels=labels.to(device)
                labels = torch.squeeze(labels, 1)
                masks = masks.to(device)
                masks = torch.squeeze(masks, 1)

                output_behrt = bert(input_ids, age_ids, gender_ids, ethnicity_ids, ins_ids, posi_ids, segment_ids,attention_mask=attMask, labels=labels)
                output_gen = generator(torch.randn(output_behrt.shape[0], noise_size).to(device))
                discr_input = torch.cat([output_behrt, output_gen], dim=0)
                features, logits, probs = discriminator(discr_input)

                features_list = torch.split(features, output_behrt.shape[0])
                D_real_features = features_list[0]
                D_fake_features = features_list[1]

                logits_list = torch.split(logits, output_behrt.shape[0])
                logits = logits_list[0]
                logits = logits[:,0:-1]

                probs_list = torch.split(probs, output_behrt.shape[0])
                D_real_probs = probs_list[0]
                D_fake_probs = probs_list[1]

                discr_loss_real = bce_loss(D_real_probs[:, -1], torch.ones(output_behrt.shape[0]).to(device))
                discr_loss_fake = bce_loss(D_fake_probs[:, -1], torch.zeros(output_behrt.shape[0]).to(device))
                discr_unsupervised_loss = (discr_loss_real + discr_loss_fake) / 2

                masked_lm_loss = bce_logits_loss(logits, labels)
                masked_lm_loss = torch.mul(masked_lm_loss, masks)
                masked_lm_loss = torch.div(masked_lm_loss.sum(dim=0), masks.sum(dim=0) + 0.001)
                discr_supervised_loss = torch.div(torch.sum(masked_lm_loss), masks.shape[1])

                discr_loss = discr_supervised_loss + train_params['alpha_unsup'] * discr_unsupervised_loss



                g_loss_d = bce_loss(D_fake_probs[:, -1], torch.ones(output_behrt.shape[0]).to(device))
                g_feat_reg = torch.mean(torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
                g_loss = g_loss_d + g_feat_reg

                optim_gen.zero_grad()
                optim_disc_behrt.zero_grad()

                g_loss.backward(retain_graph=True)
                discr_loss.backward()

                optim_gen.step()
                optim_disc_behrt.step()

                loss= g_loss.item() + discr_unsupervised_loss.item() + discr_supervised_loss.item()
                tr_loss += loss

                if step%500 == 0:
                    print("Generator Loss:", g_loss)
                    print("Discr Supervised Loss:", discr_supervised_loss)
                    print("Discr Unsupervised Loss:", discr_unsupervised_loss)

                    print("TOTAL LOSS", loss)
            cost = time.time() - start
            return tr_loss, cost
        
        def train(trainload, valload, device):
            with open("log_train.txt", 'w') as f:
                    f.write('')
            best_val = math.inf
            for e in range(train_params["epochs"]):
                print("Epoch n" + str(e))
                train_loss, train_time_cost = run_epoch(e, trainload, device)
                val_loss, val_time_cost,pred, label, mask, discr_loss = eval(valload, False, device)
                train_loss = train_loss / math.ceil((train_params["train_data_len"] * few_shots /train_params['batch_size']))
                val_loss = val_loss / math.ceil((train_params["val_data_len"]/train_params['batch_size']))
                print('TRAIN {}\t{} secs\n'.format(train_loss, train_time_cost))
                with open("log_train.txt", 'a') as f:
                    f.write("Epoch n" + str(e) + '\n TRAIN {}\t{} secs\n'.format(train_loss, train_time_cost))
                    f.write('EVAL {}\t{} secs\n'.format(val_loss, val_time_cost) + '\n\n\n')
                print('EVAL {}\t{} secs\n'.format(val_loss, val_time_cost))
                
                if discr_loss < best_val:
                    print("** ** * Saving fine - tuned model ** ** * ")
                    model_to_save = bert.module if hasattr(bert, 'module') else bert
                    save_model(model_to_save.state_dict(), f'/checkpoints/{path}/cehr-gan-bert')
                    model_to_save = generator.module if hasattr(generator, 'module') else generator
                    save_model(model_to_save.state_dict(), f'/checkpoints/{path}/generator')
                    model_to_save = discriminator.module if hasattr(discriminator, 'module') else discriminator
                    save_model(model_to_save.state_dict(), f'/checkpoints/{path}/discriminator')
                    best_val = discr_loss
                    TestDset = DataLoader(test_data, max_len=train_params['max_len_seq'], code='code')
                    testload = torch.utils.data.DataLoader(dataset=TestDset, batch_size=train_params['batch_size'], shuffle=False)
                    eval(testload, True, train_params['device'])
            return train_loss, val_loss
        
        def eval(_valload, eval, device):
            bert.eval()
            discriminator.eval()
            generator.eval()
            tr_loss = 0
            tr_d_sup = 0
            start = time.time()
            
            if eval:
                with open(f"./data/{path}/eval/custom/preds.csv", 'w') as f:
                    f.write('')
                with open(f"./data/{path}/eval/custom/labels.csv", 'w') as f:
                    f.write('')
                with open(f"./data/{path}/eval/custom/masks.csv", 'w') as f:
                    f.write('')
                    
            for step, batch in enumerate(_valload):
                batch = tuple(t for t in batch)
                input_ids, age_ids, gender_ids, ethnicity_ids, ins_ids, posi_ids, segment_ids, attMask, labels, masks = batch
                zeros = torch.zeros_like(input_ids)
                input_ids = torch.where(input_ids < zeros, zeros, input_ids)
                input_ids = input_ids.to(device)
                zeros = torch.zeros_like(age_ids)
                age_ids = torch.where(age_ids < zeros, zeros, age_ids)
                age_ids = age_ids.to(device)
                zeros = torch.zeros_like(gender_ids)
                gender_ids = torch.where(gender_ids < zeros, zeros, gender_ids)
                gender_ids = gender_ids.to(device)
                zeros = torch.zeros_like(ethnicity_ids)
                ethnicity_ids = torch.where(ethnicity_ids < zeros, zeros, ethnicity_ids)
                ethnicity_ids = ethnicity_ids.to(device)
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
                labels=labels.to(device)
                labels = torch.squeeze(labels, 1)
                masks = masks.to(device)
                masks = torch.squeeze(masks, 1)

                output_bert = bert(input_ids, age_ids, gender_ids, ethnicity_ids, ins_ids, posi_ids, segment_ids,attention_mask=attMask, labels=labels)

                features, logits, probs = discriminator(output_bert)

                logits_list = torch.split(logits, output_bert.shape[0])
                D_real_logits = logits_list[0]



                logits = D_real_logits[:,0:-1]

                masked_lm_loss = bce_logits_loss(logits, labels)
                masked_lm_loss = torch.mul(masked_lm_loss, masks)
                discr_supervised_loss = torch.div(masked_lm_loss.sum(), masks.sum() + 0.001)




                discr_loss = discr_supervised_loss 

                tr_loss += discr_loss.item()
                tr_d_sup += discr_supervised_loss.item()

                if eval:
                    with open(f"./data/{path}/eval/custom/preds.csv", 'a') as f:
                        pd.DataFrame(logits.detach().cpu().numpy()).to_csv(f, header=False)
                    with open(f"./data/{path}/eval/custom/labels.csv", 'a') as f:
                        pd.DataFrame(labels.detach().cpu().numpy()).to_csv(f, header=False)
                    with open(f"./data/{path}/eval/custom/masks.csv", 'a') as f:
                        pd.DataFrame(masks.detach().cpu().numpy()).to_csv(f, header=False)
                    
            print("Discr Supervised Loss:", tr_d_sup)

            cost = time.time() - start
            return tr_loss, cost, logits, labels, masks, tr_d_sup

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

        pretrained_dict = torch.load(f"./checkpoints/{path}/bert_pretrain", map_location=train_params['device'])
        model_dict = bert.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        bert.load_state_dict(model_dict)

        TrainDset = DataLoader(train_data, max_len=train_params['max_len_seq'], code='code')
        trainload = torch.utils.data.DataLoader(dataset=TrainDset, batch_size=train_params['batch_size'], shuffle=True)
        ValDset = DataLoader(val_data, max_len=train_params['max_len_seq'], code='code')
        valload = torch.utils.data.DataLoader(dataset=ValDset, batch_size=train_params['batch_size'], shuffle=True)
        train_loss, val_loss = train(trainload, valload, train_params['device'])