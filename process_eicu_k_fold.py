import multiprocessing
import pickle
import csv
import os
import sys
import numpy as np
import sklearn.model_selection as ms 
import torch
import time
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from layers.embedder import PackedBert
from argparse import ArgumentParser
import logging
import io
import pandas
import codecs



class EncounterInfo:
    def __init__(self, patient_id, encounter_id, 
                 encounter_timestamp, expired, readmission, label_los=-1.0):
        self.patient_id = patient_id
        self.encounter_id = encounter_id
        self.encounter_timestamp = encounter_timestamp
        self.expired = expired
        self.readmission = readmission
        self.dx_ids = []
        self.rx_ids = []
        self.labs = {}

        self.treatments = []
        self.label_los=label_los

class EncounterFeatures:
    def __init__(self, patient_id, label_expired, label_readmission, dx_ids, dx_ints, proc_ids, proc_ints, label_los):
        self.patient_id = patient_id
        self.label_expired = label_expired
        self.label_readmission = label_readmission
        self.dx_ids = dx_ids
        self.dx_ints = dx_ints
        self.proc_ids = proc_ids
        self.proc_ints = proc_ints
        self.label_los = label_los
        self.prior_indices = None
        self.prior_values = None
        self.kg_prior_indices = None
        self.kg_prior_values = None
        self.bert_prior_indices = None
        self.bert_prior_values = None
        self.dx_mask = None
        self.proc_mask = None
        

def process_patient(infile, encounter_dict, hour_threshold=24):
   
    with open(infile, 'r',encoding='utf-8-sig') as f:
        count = 0
        for line in tqdm(csv.DictReader(f)):

            patient_id = line['patienthealthsystemstayid']
            encounter_id = line['patientunitstayid']

            encounter_timestamp = float(line['hospitaladmitoffset'])
            discharge_status = line['unitdischargestatus']
            end_timestamp = float(line['unitdischargeoffset'])
            duration_day = (end_timestamp - encounter_timestamp) / (60 * 24)

            expired = True if discharge_status=='Expired' else False
            readmission = 1 

            if duration_day > 60.: 
                continue
            if duration_day < 0:
                continue
            ei = EncounterInfo(patient_id, encounter_id, encounter_timestamp, expired, readmission, duration_day)
            
            if encounter_id in encounter_dict:
                logging.info('duplicate encounter id! skip')
                sys.exit(0)
            encounter_dict[encounter_id] = ei
            count += 1
    return encounter_dict

def process_admission_dx(infile, encounter_dict):
    
    with open(infile, 'r') as f:
        count = 0
        missing_eid = 0
        for line in tqdm(csv.DictReader(f)):
            encounter_id = line['patientunitstayid']
            dx_id = line['admitdxpath'].lower()
            
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue
            encounter_dict[encounter_id].dx_ids.append(dx_id)
            count += 1
    logging.info('')
    logging.info('Admission Diagnosis without encounter id: {}'.format(missing_eid))
    return encounter_dict

def process_diagnosis(infile, encounter_dict):

    with open(infile, 'r', encoding='utf-8-sig') as f:
        count = 0
        missing_eid = 0
        for line in tqdm(csv.DictReader(f)):
            encounter_id = line['patientunitstayid']
            dx_id = line['diagnosisstring'].lower()
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue
            encounter_dict[encounter_id].dx_ids.append(dx_id)
            count += 1
    logging.info('Diagnosis without encounter id: {}'.format(missing_eid))
    return encounter_dict

def process_treatment(infile, encounter_dict):

    with open(infile, 'r', encoding='utf-8-sig') as f:
        count = 0
        missing_eid = 0
        for line in tqdm(csv.DictReader(f)):
            encounter_id = line['patientunitstayid']
            treatment_id = line['treatmentstring'].lower()
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue
            encounter_dict[encounter_id].treatments.append(treatment_id)
            count += 1

    logging.info('Treatment without encounter id: {}'.format(missing_eid))
    logging.info('accepted treatment: {}'.format(count))
    return encounter_dict

def process_apache(infile, encounter_dict):
    with open(infile, 'r') as f:
        count = 0
        missing_eid = 0
        missing_los = 0
        for line in tqdm(csv.DictReader(f)):
            encounter_id = line['patientunitstayid']
            los = line['actualhospitallos']
            if encounter_id not in encounter_dict:
                missing_eid += 1
                continue
            if float(los) < 0:
                missing_los += 1 
                del encounter_dict[encounter_id]
                logging.info(f'encounter_id is {encounter_id}, los is smaller than 0!')
                continue
            encounter_dict[encounter_id].label_los = float(los)
            count += 1
    logging.info('Apache without encounter id: {}'.format(missing_eid))
    logging.info('Apache without los: {}'.format(missing_los))
    logging.info('accepted Apache: {}'.format(count))
    return encounter_dict


def get_encounter_features(encounter_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=200):

    key_list = []
    enc_features_list = []
    dx_str2int = {}
    treat_str2int = {}
    dx_int2str = {}
    treat_int2str = {}
    num_cut = 0
    num_duplicate = 0
    count = 0
    num_dx_ids = 0
    num_treatments = 0
    num_unique_dx_ids = 0
    num_unique_treatments = 0
    min_dx_cut = 0
    min_treatment_cut = 0
    max_dx_cut = 0
    max_treatment_cut = 0
    num_expired = 0
    num_readmission = 0
    num_los = 0
    

    
    for _, enc in tqdm(encounter_dict.items()):
        if skip_duplicate:
            if (len(enc.dx_ids) > len(set(enc.dx_ids)) or len(enc.treatments) > len(set(enc.treatments))):
                num_duplicate += 1
                continue
        if len(set(enc.dx_ids)) < min_num_codes:
            min_dx_cut += 1
            continue
        if len(set(enc.treatments)) < min_num_codes:
            min_treatment_cut += 1
            continue
        if len(set(enc.dx_ids)) > max_num_codes:
            max_dx_cut += 1
            continue
        if len(set(enc.treatments)) > max_num_codes:
            max_treatment_cut += 1
            continue
        
        count += 1
        num_dx_ids += len(enc.dx_ids)
        num_treatments += len(enc.treatments)
        num_unique_dx_ids += len(set(enc.dx_ids))
        num_unique_treatments += len(set(enc.treatments))

        for dx_id in enc.dx_ids:
            if dx_id not in dx_str2int:
                dx_str2int[dx_id] = len(dx_str2int)
                dx_int2str[str(len(dx_str2int))] = dx_id
        for treat_id in enc.treatments:
            if treat_id not in treat_str2int:
                treat_str2int[treat_id] = len(treat_str2int)
                treat_int2str[str(len(treat_str2int))] = treat_id
        
        patient_id = enc.patient_id + ':' + enc.encounter_id
        if enc.expired:
            label_expired = 1
            num_expired += 1
            # TODO
            logging.info(f'encounter_id is {enc.encounter_id}, this patient is expired')
            continue
        else:
            label_expired = 0
        if enc.readmission:
            label_readmission = 1
            num_readmission += 1
        else:
            label_readmission = 0
        
        dx_ids = sorted(list(set(enc.dx_ids)))
        dx_ints = [dx_str2int[item] for item in dx_ids]
        proc_ids = sorted(list(set(enc.treatments)))
        proc_ints = [treat_str2int[item] for item in proc_ids]
        
        if enc.label_los == -1.0:
            logging.info(f'encounter_id is {enc.encounter_id}, los is missing in actialhospitallos csv file')
            continue
        num_los += 1
        label_los = enc.label_los

        
        enc_features = EncounterFeatures(patient_id, label_expired, label_readmission, dx_ids, dx_ints, proc_ids, proc_ints, label_los)
        
        key_list.append(patient_id)
        enc_features_list.append(enc_features)
    
    
    for ef in enc_features_list:
        dx_padding_idx = len(dx_str2int)
        proc_padding_idx = len(treat_str2int)
        if len(ef.dx_ints) < max_num_codes:
            ef.dx_ints.extend([dx_padding_idx]*(max_num_codes-len(ef.dx_ints)))
        if len(ef.proc_ints) < max_num_codes:
            ef.proc_ints.extend([proc_padding_idx]*(max_num_codes-len(ef.proc_ints)))
        ef.dx_mask = [0 if i==dx_padding_idx else 1 for i in ef.dx_ints]
        ef.proc_mask = [0 if i==proc_padding_idx else 1 for i in ef.proc_ints]

        

    logging.info('Filtered encounters due to duplicate codes: %d' % num_duplicate)
    logging.info('Filtered encounters due to thresholding: %d' % num_cut)
    logging.info('Average num_dx_ids: %f' % (num_dx_ids / count))
    logging.info('Average num_treatments: %f' % (num_treatments / count))
    logging.info(f'vocab size of dx_ids is {len(dx_str2int)}')
    logging.info(f'vocab size of proc_ids is {len(treat_str2int)}')
    logging.info('Average num_unique_dx_ids: %f' % (num_unique_dx_ids / count))
    logging.info('Average num_unique_treatments: %f' % (num_unique_treatments / count))

    logging.info('Min dx cut: %d' % min_dx_cut)
    logging.info('Min treatment cut: %d' % min_treatment_cut)
    logging.info('Max dx cut: %d' % max_dx_cut)
    logging.info('Max treatment cut: %d' % max_treatment_cut)
    logging.info('Number of expired: %d' % num_expired)
    logging.info('Number of readmission: %d' % num_readmission)
    logging.info('Number of hospitallos: %d' % num_los)


    return key_list, enc_features_list, dx_str2int, treat_str2int, dx_int2str, treat_int2str



def select_train_valid_test(key_list, random_seed=1234):

    kf = ms.KFold(n_splits=5, shuffle=True, random_state=random_seed)
    return kf.split(key_list)


def count_conditional_prob_dp(enc_features_list, output_path, train_key_set=None):
    dx_freqs = {}
    proc_freqs = {}
    dp_freqs = {}
    total_visit = 0
    for enc_feature in tqdm(enc_features_list):
        key = enc_feature.patient_id
        if (train_key_set is not None and key not in train_key_set):
            total_visit += 1
            continue
        dx_ids = enc_feature.dx_ids
        proc_ids = enc_feature.proc_ids
        for dx in dx_ids:
            if dx not in dx_freqs:
                dx_freqs[dx] = 0
            dx_freqs[dx] += 1
        for proc in proc_ids:
            if proc not in proc_freqs:
                proc_freqs[proc] = 0
            proc_freqs[proc] += 1
        for dx in dx_ids:
            for proc in proc_ids:
                dp = dx + ',' + proc
                if dp not in dp_freqs: 
                    dp_freqs[dp] = 0
                dp_freqs[dp] += 1
        total_visit += 1
    
    dx_probs = dict([(k, v / float(total_visit)) for k, v in dx_freqs.items()
                ])
    proc_probs = dict([
    (k, v / float(total_visit)) for k, v in proc_freqs.items()
    ])
    dp_probs = dict([(k, v / float(total_visit)) for k, v in dp_freqs.items()
                ])

          
    dis_med_df = pandas.read_csv('dataset/hf0812_los/disease_treatment.csv')
    dis_med_set = set(map(lambda x: f'{x[0]},{x[1]}', list(dis_med_df.values)))
    dis_count = {k:v for k, v in map(lambda x: (x[0], len(x[1])), list(dis_med_df.groupby('disease')))}
    med_count = {k:v for k, v in map(lambda x: (x[0], len(x[1])), list(dis_med_df.groupby('med')))}


    dp_cond_probs = {}
    pd_cond_probs = {}
    dis_med_probs = {}
    med_dis_probs = {}
    for dx, dx_prob in dx_probs.items():
        for proc, proc_prob in proc_probs.items():
            dp = dx + ',' + proc
            pd = proc + ',' + dx
            if dp in dp_probs:
                dp_cond_probs[dp] = dp_probs[dp] / dx_prob
                pd_cond_probs[pd] = dp_probs[dp] / proc_prob
            else:
                dp_cond_probs[dp] = 0.0
                pd_cond_probs[pd] = 0.0
            if dp in dis_med_set:
                dis_med_probs[dp] = 1/dis_count[dx]
                med_dis_probs[pd] = 1/med_count[proc]




    pickle.dump(dp_cond_probs, open(os.path.join(output_path, 'dp_cond_probs.empirical.p'), 'wb'))
    pickle.dump(pd_cond_probs, open(os.path.join(output_path, 'pd_cond_probs.empirical.p'), 'wb'))

    pickle.dump(dis_med_probs, open(os.path.join('dataset/hf0812_los', 'dp_cond_probs.kb.p'), 'wb'))
    pickle.dump(med_dis_probs, open(os.path.join('dataset/hf0812_los', 'pd_cond_probs.kb.p'), 'wb'))



def count_conditional_prob_dp_same_type(enc_features_list, output_path, train_key_set=None):
    dx_freqs = {}
    proc_freqs = {}
    dp_freqs = {}


    total_visit = 0
    for enc_feature in tqdm(enc_features_list):
        key = enc_feature.patient_id
        if (train_key_set is not None and key not in train_key_set):
            total_visit += 1
            continue
        dx_ids = enc_feature.dx_ids
        proc_ids = enc_feature.proc_ids
        
        for dx in dx_ids:
            if dx not in dx_freqs:
                dx_freqs[dx] = 0
            dx_freqs[dx] += 1
        
        for proc in proc_ids:
            if proc not in proc_freqs:
                proc_freqs[proc] = 0
            proc_freqs[proc] += 1

        
        for dx in dx_ids:
            for proc in proc_ids:
                dp = dx + ',' + proc
                if dp not in dp_freqs: 
                    dp_freqs[dp] = 0
                dp_freqs[dp] += 1
        total_visit += 1

        for dx in dx_ids:
            for 
    
    dx_probs = dict([(k, v / float(total_visit)) for k, v in dx_freqs.items()
                ])
    proc_probs = dict([
    (k, v / float(total_visit)) for k, v in proc_freqs.items()
    ])
    dp_probs = dict([(k, v / float(total_visit)) for k, v in dp_freqs.items()
                ])

    # start: count total amount of disease and medicine            
    dis_med_df = pandas.read_csv('dataset/hf0812_los/disease_treatment.csv')
    dis_med_set = set(map(lambda x: f'{x[0]},{x[1]}', list(dis_med_df.values)))
    dis_count = {k:v for k, v in map(lambda x: (x[0], len(x[1])), list(dis_med_df.groupby('disease')))}
    med_count = {k:v for k, v in map(lambda x: (x[0], len(x[1])), list(dis_med_df.groupby('med')))}
    # end: count total amount of disease and medicine

    dp_cond_probs = {}
    pd_cond_probs = {}
    dis_med_probs = {}
    med_dis_probs = {}
    for dx, dx_prob in dx_probs.items():
        for proc, proc_prob in proc_probs.items():
            dp = dx + ',' + proc
            pd = proc + ',' + dx
            if dp in dp_probs:
                dp_cond_probs[dp] = dp_probs[dp] / dx_prob
                pd_cond_probs[pd] = dp_probs[dp] / proc_prob
            else:
                dp_cond_probs[dp] = 0.0
                pd_cond_probs[pd] = 0.0
            if dp in dis_med_set:
                dis_med_probs[dp] = 1/dis_count[dx]
                med_dis_probs[pd] = 1/med_count[proc]

    pickle.dump(dp_cond_probs, open(os.path.join(output_path, 'dp_cond_probs.empirical.p'), 'wb'))
    pickle.dump(pd_cond_probs, open(os.path.join(output_path, 'pd_cond_probs.empirical.p'), 'wb'))

    pickle.dump(dis_med_probs, open(os.path.join('dataset/hf0812_los', 'dp_cond_probs.kb.p'), 'wb'))
    pickle.dump(med_dis_probs, open(os.path.join('dataset/hf0812_los', 'pd_cond_probs.kb.p'), 'wb'))




def add_sparse_prior_guide_dp(enc_features_list, stats_path, key_set=None, max_num_codes=200):
    


    dp_cond_probs = pickle.load(open(os.path.join(stats_path, 'dp_cond_probs.empirical.p'), 'rb'))
    pd_cond_probs = pickle.load(open(os.path.join(stats_path, 'pd_cond_probs.empirical.p'), 'rb'))
    bert_dp_cond_probs = pickle.load(open(os.path.join('dataset/hf0812_los', 'dp_cond_probs.nsp.p'), 'rb'))
    bert_pd_cond_probs = pickle.load(open(os.path.join('dataset/hf0812_los', 'pd_cond_probs.nsp.p'), 'rb'))
    kg_dp_cond_probs = pickle.load(open(os.path.join('dataset/hf0812_los', 'dp_cond_probs.kb.p'), 'rb'))
    kg_pd_cond_probs = pickle.load(open(os.path.join('dataset/hf0812_los', 'pd_cond_probs.kb.p'), 'rb'))
    total_visit = 0
    new_enc_features_list = []

    for enc_features in enc_features_list:
        key = enc_features.patient_id
        if (key_set is not None and key not in key_set):
            total_visit += 1
            continue
        dx_ids = enc_features.dx_ids
        proc_ids = enc_features.proc_ids
        indices = []
        values = []
        bert_indices = []
        bert_values = []
        kg_indices = []
        kg_values = []
        for i, dx in enumerate(dx_ids):
            for j, proc in enumerate(proc_ids):
                dp = dx + ',' + proc
                indices.append((i, max_num_codes+j))
                bert_indices.append((i, max_num_codes+j))
                kg_indices.append((i, max_num_codes+j))
                prob = 0.0 if dp not in dp_cond_probs else dp_cond_probs[dp]
                values.append(prob)
                prob = 0.0 if dp not in bert_dp_cond_probs else bert_dp_cond_probs[dp]
                bert_values.append(prob)
                prob = 0.0 if dp not in kg_dp_cond_probs else kg_dp_cond_probs[dp]
                kg_values.append(prob)
        for i, proc in enumerate(proc_ids):
            for j, dx in enumerate(dx_ids):
                pd = proc + ',' + dx
                indices.append((max_num_codes+i, j))
                prob = 0.0 if pd not in pd_cond_probs else pd_cond_probs[pd]
                values.append(prob)

                bert_indices.append((max_num_codes+i, j))
                prob = 0.0 if pd not in bert_pd_cond_probs else bert_pd_cond_probs[pd]
                bert_values.append(prob)

                kg_indices.append((max_num_codes+i, j))
                prob = 0.0 if pd not in kg_pd_cond_probs else kg_pd_cond_probs[pd]
                kg_values.append(prob)



        enc_features.prior_indices = indices
        enc_features.prior_values = values

        enc_features.bert_prior_indices = bert_indices
        enc_features.bert_prior_values = bert_values

        enc_features.kg_prior_indices = kg_indices
        enc_features.kg_prior_values = kg_values

        new_enc_features_list.append(enc_features)
    

        total_visit += 1
    return new_enc_features_list
        
def convert_features_to_tensors(enc_features):

    all_readmission_labels = torch.tensor([f.label_readmission for f in enc_features], dtype=torch.long)
    all_expired_labels = torch.tensor([f.label_expired for f in enc_features], dtype=torch.long)
    all_dx_ints = torch.tensor([f.dx_ints for f in enc_features], dtype=torch.long)
    all_proc_ints = torch.tensor([f.proc_ints for f in enc_features], dtype=torch.long)

    all_dx_masks = torch.tensor([f.dx_mask for f in enc_features], dtype=torch.float)
    all_proc_masks = torch.tensor([f.proc_mask for f in enc_features], dtype=torch.float)
    all_los = torch.tensor([f.label_los for f in enc_features], dtype=torch.float)
    dataset = TensorDataset(all_dx_ints, all_proc_ints, all_dx_masks, all_proc_masks, all_readmission_labels, all_expired_labels, all_los)
    
    
    return dataset


def get_prior_guide(enc_features, prior_type='co_occur'):

    prior_guide_list = []
    if prior_type == 'co_occur':
        for feats in enc_features:
            indices = torch.tensor(list(zip(*feats.prior_indices))).reshape(2, -1)
            values = torch.tensor(feats.prior_values)
            prior_guide_list.append((indices, values))
    elif prior_type == 'bert':
        for feats in enc_features:
            indices = torch.tensor(list(zip(*feats.bert_prior_indices))).reshape(2, -1)
            values = torch.tensor(feats.bert_prior_values)
            prior_guide_list.append((indices, values))

    elif prior_type =='kg':
        for feats in enc_features:
            indices = torch.tensor(list(zip(*feats.kg_prior_indices))).reshape(2, -1)
            values = torch.tensor(feats.kg_prior_values)
            prior_guide_list.append((indices, values))
    else:
        assert False, F'prior type error, it must be co_occur, bert or kg, but now is {prior_type}'
    return prior_guide_list


def get_datasets(data_dir, prior_type, fold=0):
    #instead of generating 5 folds manually prior to training using 2 separate scripts, let's generate 1 fold in same script
    patient_file = os.path.join(data_dir, 'patient.csv')
    diagnosis_file = os.path.join(data_dir, 'diagnosis.csv')
    treatment_file = os.path.join(data_dir, 'treatment.csv')

    fold_path = os.path.join(data_dir, 'fold_{}'.format(fold))
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    stats_path = os.path.join(fold_path, 'train_stats')
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
    cached_path = os.path.join(fold_path, 'cached')

    if os.path.exists(cached_path):
        start = time.time()
        train_dataset = torch.load(os.path.join(cached_path, 'train_dataset.pt'))
        validation_dataset = torch.load(os.path.join(cached_path, 'valid_dataset.pt'))
        test_dataset = torch.load(os.path.join(cached_path, 'test_dataset.pt'))

        # co_occur prior
        train_prior_guide = torch.load(os.path.join(cached_path, 'train_priors.pt'))
        validation_prior_guide = torch.load(os.path.join(cached_path, 'valid_priors.pt'))
        test_prior_guide = torch.load(os.path.join(cached_path, 'test_priors.pt'))

        # bert prior
        bert_train_prior_guide = torch.load(os.path.join(cached_path, 'train_priors_bert.pt'))
        bert_validation_prior_guide = torch.load(os.path.join(cached_path, 'valid_priors_bert.pt'))
        bert_test_prior_guide = torch.load(os.path.join(cached_path, 'test_priors_bert.pt'))
        
        # knowledge graph prior
        kg_train_prior_guide = torch.load(os.path.join(cached_path, 'train_priors_kg.pt'))
        kg_validation_prior_guide = torch.load(os.path.join(cached_path, 'valid_priors_kg.pt'))
        kg_test_prior_guide = torch.load(os.path.join(cached_path, 'test_priors_kg.pt'))
        
    else:
        MAX_NUM_CODES = 200
        encounter_dict = {}
        logging.info('Processing patient.csv')
        encounter_dict = process_patient(patient_file, encounter_dict, hour_threshold=24)
        logging.info('Processing diagnosis.csv')
        encounter_dict = process_diagnosis(diagnosis_file, encounter_dict)
        logging.info('Processing treatment.csv')
        encounter_dict = process_treatment(treatment_file, encounter_dict)

        
        key_list, enc_features_list, dx_map, proc_map, dx_int2str, treat_int2str = get_encounter_features(encounter_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=MAX_NUM_CODES)

        pickle.dump(dx_map, open(os.path.join(data_dir, 'dx_map.p'), 'wb'))
        pickle.dump(proc_map, open(os.path.join(data_dir, 'proc_map.p'), 'wb'))
        pickle.dump(dx_int2str, open(os.path.join(data_dir, 'dx_int2str.p'), 'wb'))
        pickle.dump(treat_int2str, open(os.path.join(data_dir, 'treat_int2str.p'), 'wb'))
 
        packedBert = PackedBert(map_path=data_dir, device='cuda')
        packedBert.count_dp_cond_probs(enc_features_list, data_dir)
        
        for fold, train_and_test in enumerate(select_train_valid_test(key_list, random_seed=fold)):
            
            fold_path = os.path.join(data_dir, 'fold_{}'.format(fold))
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)
            stats_path = os.path.join(fold_path, 'train_stats')
            if not os.path.exists(stats_path):
                os.makedirs(stats_path)
            cached_path = os.path.join(fold_path, 'cached')
            if not os.path.exists(cached_path):
                os.makedirs(cached_path)
  
  
            key_train, key_temp = train_and_test
            key_valid, key_test = ms.train_test_split(key_temp, test_size=0.5)
            key_train, key_test, key_valid = np.array(key_list)[key_train].tolist(), np.array(key_list)[key_test].tolist(), np.array(key_list)[key_valid].tolist()
            count_conditional_prob_dp(enc_features_list, stats_path, set(key_train))


            logging.info(f'total amount of data is {len(enc_features_list)}')
            
            train_enc_features = add_sparse_prior_guide_dp(enc_features_list, stats_path, set(key_train), max_num_codes=MAX_NUM_CODES)
            validation_enc_features = add_sparse_prior_guide_dp(enc_features_list, stats_path, set(key_valid), max_num_codes=MAX_NUM_CODES)
            test_enc_features = add_sparse_prior_guide_dp(enc_features_list, stats_path, set(key_test), max_num_codes=MAX_NUM_CODES)
            logging.info(f'total amount of data for training is {len(train_enc_features)}')
            logging.info(f'total amount of data for validating is {len(validation_enc_features)}')
            logging.info(f'total amount of data for testing is {len(test_enc_features)}')
            
            train_dataset = convert_features_to_tensors(train_enc_features)
            validation_dataset = convert_features_to_tensors(validation_enc_features)
            test_dataset = convert_features_to_tensors(test_enc_features)
            
            torch.save(train_dataset, os.path.join(cached_path, 'train_dataset.pt'))
            torch.save(validation_dataset, os.path.join(cached_path, 'valid_dataset.pt'))
            torch.save(test_dataset, os.path.join(cached_path, 'test_dataset.pt'))
            
            ## get prior_indices and prior_values for each split and save as list of tensors
            train_prior_guide = get_prior_guide(train_enc_features)
            validation_prior_guide = get_prior_guide(validation_enc_features)
            test_prior_guide = get_prior_guide(test_enc_features)
            
            #save the prior_indices and prior_values
            torch.save(train_prior_guide, os.path.join(cached_path, 'train_priors.pt'))
            torch.save(validation_prior_guide, os.path.join(cached_path, 'valid_priors.pt'))
            torch.save(test_prior_guide, os.path.join(cached_path, 'test_priors.pt'))


            bert_train_prior_guide = get_prior_guide(train_enc_features, 'bert')
            bert_validation_prior_guide = get_prior_guide(validation_enc_features, 'bert')
            bert_test_prior_guide = get_prior_guide(test_enc_features, 'bert')
            
            # save the prior_indices and prior_values
            torch.save(bert_train_prior_guide, os.path.join(cached_path, 'train_priors_bert.pt'))
            torch.save(bert_validation_prior_guide, os.path.join(cached_path, 'valid_priors_bert.pt'))
            torch.save(bert_test_prior_guide, os.path.join(cached_path, 'test_priors_bert.pt'))      

            # # generate kb prior
            kg_train_prior_guide = get_prior_guide(train_enc_features, 'kg')
            kg_validation_prior_guide = get_prior_guide(validation_enc_features, 'kg')
            kg_test_prior_guide = get_prior_guide(test_enc_features, 'kg')

            torch.save(kg_train_prior_guide, os.path.join(cached_path, 'train_priors_kg.pt'))
            torch.save(kg_validation_prior_guide, os.path.join(cached_path, 'valid_priors_kg.pt'))
            torch.save(kg_test_prior_guide, os.path.join(cached_path, 'test_priors_kg.pt'))     

        
    
    
    return ([train_dataset, validation_dataset, test_dataset], 
    [train_prior_guide, validation_prior_guide, test_prior_guide], 
    [bert_train_prior_guide, bert_validation_prior_guide, bert_test_prior_guide],
    [kg_train_prior_guide, kg_validation_prior_guide, kg_test_prior_guide])
        

if __name__ == '__main__':

    logdir = os.getcwd()
    if not os.path.exists(os.path.join(logdir, f'logs')):
        os.mkdir(os.path.join(logdir, f'logs'))
    logging.basicConfig(
        filename=os.path.join(logdir, f'logs/process_icu.log'),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logging.info('***** Processing eicu *****')

    data_dir = 'dataset/hf0812_los'
    logging.info(f'data_dir: {data_dir}')

    get_datasets(data_dir, prior_type='co_occur')
        
        
        
        
        
        
        
    
        
        
        
    
    

    
    