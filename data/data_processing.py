## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14


import json
import torch
import os
import pickle
import numpy as np
from copy import copy, deepcopy
from tqdm import tqdm
from loguru import logger
from util.util import *
import random 
from data.SingleDataset import SingleDataset
from sklearn.model_selection import train_test_split

import itertools

class DatasetSISSF():
    

    def __init__(self, special_token_idx, token_freq_th, weight_th, context_truncate,response_truncate, dpath, dtype , mode):
        """

       special_token_idx: {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'pad_entity': 0,
            'pad_word': 0
        },

        """
        
        super().__init__()
        self.special_token_idx = special_token_idx
        self.unk_token_idx = self.special_token_idx['unk']
        self.token_freq_th = token_freq_th
        self.weight_th = weight_th
        self.dpath = dpath
        self.dtype = dtype 
        self.mode = mode
        self.context_truncate = context_truncate
        # self.item_truncate = item_truncate
        self.response_truncate = response_truncate
        self.load_data()
        self.data_preprocess()
        

  
        
    
    def load_data(self):
        # load train/valid/test data
        print(f"Dataset: {self.dtype}")
        
        with open(os.path.join(self.dpath,self.dtype, 'train_data.json'), 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)

        with open(os.path.join(self.dpath,self.dtype, 'valid_data.json'), 'r', encoding='utf-8') as f:
            self.valid_data = json.load(f)
        with open(os.path.join(self.dpath,self.dtype, 'test_data.json'), 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        self.tok2ind = json.load(open(os.path.join(self.dpath, self.dtype,'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}    
        if self.dtype == 'ReDial':
            path_data = self.dpath + "/" +  self.dtype + "/redial_interactions.pickle"
        else :
            path_data = self.dpath + "/" +  self.dtype + "/inspired_interactions.pickle"    
        data_file = open(path_data, 'rb')
        self.history_u_lists, self.history_ur_lists, self.history_v_lists, self.history_vr_lists,self.train_u,self.train_v, self.train_r, self.test_u, self.test_v, self.test_r, self.social_adj_lists, self.social_adj_ratings, self.ratings_list = pickle.load(data_file)

        
        conv_tokID2freq = dict(json.load(open(os.path.join(self.dpath, self.dtype, 'token_freq.json'))))
        self.decoder_token_prob_weight = self.get_decoder_decoder_token_prob_weight(conv_tokID2freq)
        self.user_mapping = {}
        self.item_mapping = {}
        self.users = []
        self.items = []
        with open(os.path.join(self.dpath, self.dtype, 'user_list.txt')) as f:
            check = False
            for l in f.readlines():
                if check :
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        uid = int(l[0])
                        user_id = int(l[1])
                        self.user_mapping[uid] = user_id 
                        self.users.append(user_id)
                else : 
                    check = True         

        with open(os.path.join(self.dpath,self.dtype, 'item_list.txt')) as f:
            check = False
            for l in f.readlines():
                if check :
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        iid = int(l[0])
                        item_id = int(l[1])
                        self.item_mapping[iid] = item_id
                        self.items.append(item_id)
                        
                else : 
                    check = True 

        self.n_users = self.history_u_lists.__len__()
        self.n_items = self.history_v_lists.__len__()
        self.num_ratings = self.ratings_list.__len__()
        self.vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            'vocab_size': len(self.tok2ind),
            'n_user' : self.n_users,
            'n_items' : self.n_items,
            'user_mapping' : self.user_mapping,
            'item_mapping' : self.item_mapping,
            'decoder_token_prob_weight': self.decoder_token_prob_weight
        }
        self.vocab.update(self.special_token_idx)    

        

    
    def get_decoder_decoder_token_prob_weight(self, conv_tokID2freq):
        decoder_token_prob_weight = []
        nb_reform = 0
        
        for tokID in range(max(list(self.tok2ind.values())) + 1):   
            freq = conv_tokID2freq.get(tokID, 1)
            weight = (self.token_freq_th * 1.0) / freq if freq > self.token_freq_th else 1.0
            reform_weight = max(self.weight_th, weight)
            if reform_weight != weight:
                nb_reform += 1
            decoder_token_prob_weight.append(reform_weight)

        decoder_token_prob_weight = torch.FloatTensor(decoder_token_prob_weight) # (nb_tok)
        decoder_token_prob_weight = decoder_token_prob_weight.unsqueeze(0).unsqueeze(0) # (1, 1, nb_tok)

        return decoder_token_prob_weight

  
    def data_preprocess(self):
        
        
        self.train_data =SingleDataset(self._raw_data_process( self.train_data , "Processing Train Dataset"))
        self.valid_data = SingleDataset(self._raw_data_process( self.valid_data , "Processing Valid Dataset"))
        self.test_data = SingleDataset(self._raw_data_process( self.test_data , "Processing Test Dataset")) 
    

       
        
        

    def _raw_data_process(self, raw_data, mode):
        logger.info(mode)
        augmented_conv = self.merge_conv_data_add_entities_mask(raw_data)
        augmented_conv = self.add_item_context(augmented_conv)
        # augmented_conv = self.augment_item_context(augmented_conv)
        augmented_conv = self.augment_and_add_add_entities_mask(augmented_conv)
        augmented_conv = self.seperate_rec_items(augmented_conv)
        # augmented_conv = self.augment_nlp_context(augmented_conv)
        return augmented_conv

    def _raw_data_process_test(self, raw_data, mode):
        logger.info(mode)
        augmented_conv = self.merge_conv_data_add_entities_mask(raw_data)
        print(augmented_conv)
        augmented_conv = self.add_item_context(augmented_conv)
        print(augmented_conv)
        augmented_conv = self.spec_augment_and_add_add_entities_mask(augmented_conv)
        print(augmented_conv)
        augmented_conv = self.spec_seperate_rec_items(augmented_conv)
        print(augmented_conv)
        return augmented_conv    
    
    def _raw_data_process_no_shfting(self, raw_data, mode):
        logger.info(mode)
        augmented_conv = self.merge_conv_data_add_entities_mask(raw_data)
        augmented_conv = self.augment_and_add_add_entities_mask(augmented_conv)
        augmented_conv = self.seperate_rec_items(augmented_conv)
        return augmented_conv



    def _raw_data_process_reduce(self, raw_data, mode):
        logger.info(mode)
        augmented_conv = self.merge_conv_data_add_entities_mask(raw_data, True)
        augmented_conv = self.augment_and_add_add_entities_mask(augmented_conv)
        # augmented_conv = self.augment_and_shifting(augmented_conv)
        augmented_conv = self.seperate_rec_items(augmented_conv)
        return augmented_conv
    

    def merge_conv_data_add_entities_mask(self, conversations, reduce=False):
        logger.info("Merge Conversations")
        if reduce:
            # Calculate 30% of the length of the list
            thirty_percent = int(len(conversations) * 0.05)

            # Take the first 30% of the list
            conversations = conversations[:thirty_percent]  
        for conversation in tqdm(conversations):
            augmented_messages = []
            last_user = None
          
            for utt in conversation['messages']:
                text_token_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in utt["text"]]
                rec_item_ids = [item for item in utt['rec_items']]
                items = [item for item in utt['items'] if item not in rec_item_ids]
                entity_ids = [entity for entity in utt['entities']]
                item_ids_in_context, items_mask_in_context, item_masks_in_context = self.get_items_info_in_context_text(utt, text_token_ids)
           
                if utt["senderWorkerId"] == last_user:
                    augmented_messages[-1]["text"] += text_token_ids # [utter_len]
                    augmented_messages[-1]["rec_items"].extend(rec_item_ids),
                    augmented_messages[-1]["items"].extend(items),
                    augmented_messages[-1]["entities"] += entity_ids
                    augmented_messages[-1]["items_mask_in_context"] += items_mask_in_context # [utter_len]
                    augmented_messages[-1]["item_masks_in_context"] += item_masks_in_context # [n_items_in_utter_text, utter_len]
                    augmented_messages[-1]["item_ids_in_context"] += item_ids_in_context # [n_items_in_utter_text]
                else:
                    augmented_messages.append({
                        "senderWorkerId": utt["senderWorkerId"],
                        "text":  text_token_ids, # [utter_len][int(self.tok2ind[id])] + text_token_ids
                        "items": items,
                        "rec_items": rec_item_ids,
                        "entities": entity_ids,
                        'items_mask_in_context': items_mask_in_context, # [utter_len]
                        'item_masks_in_context': item_masks_in_context, # [n_items_in_utter_text, utter_len]
                        'item_ids_in_context': item_ids_in_context, # [n_items_in_utter_text]
                    })
                    
                   

                last_user = utt["senderWorkerId"]
               
            conversation['messages']= augmented_messages  
            itemsALl = set() 
        

        return conversations

    def add_item_context(self, conversations, reduce=False):
        logger.info("Items Cotext Conversations")
        if reduce:
            # Calculate 30% of the length of the list
            thirty_percent = int(len(conversations) * 0.05)

            # Take the first 30% of the list
            conversations = conversations[:thirty_percent]  
        for conversation in tqdm(conversations):
            augmented_messages = []
            context_item= set()
            for utt in conversation['messages']:
                rec_item_ids =  utt['rec_items']
                
    
               
                augmented_messages.append({
                    "senderWorkerId": utt["senderWorkerId"],
                    "text":  utt["text"], # [utter_len][int(self.tok2ind[id])] + text_token_ids
                    "items": list(context_item),
                    "rec_items": rec_item_ids,
                    "entities": utt["entities"],
                    'items_mask_in_context': utt["items_mask_in_context"], # [utter_len]
                    'item_masks_in_context': utt["item_masks_in_context"], # [n_items_in_utter_text, utter_len]
                    'item_ids_in_context': utt["item_ids_in_context"], # [n_items_in_utter_text]
                })
                    
                   

                context_item.update(rec_item_ids)
            conversation['messages']= augmented_messages  
           
        

        return conversations   
    
    def augment_item_context(self, conversations): 
        allConv = [] 
        for conversation in tqdm(conversations):
            augmented_messages = []
            conver = {}
            context_item= set()
            for index, utt in enumerate(conversation['messages']):
                items =  utt['items']
                if len(items) > 1:
                    all_combinations = []
                    for r in range(1, len(items) + 1):
                        if random.random() < 0.1:  # 1% chance
                            combinations = itertools.combinations(items, r)
                            all_combinations.extend(combinations)
                        else:
                            continue
                            

                    # Print all combinations
                    for order, combo in enumerate(all_combinations):
                        # print(combo)
                        uttx = deepcopy(utt)
                        uttx['items'] = list(combo)
                        if order == 0:
                            conver[index] = [uttx]
                        else:
                            conver[index].append(uttx)     
                        # augmented_messages.append(conversation)
                       
                augmented_messages.append(utt)          
            conversation['messages']= augmented_messages  
            allConv.append(conversation)
            for index,conPair in conver.items():
                for item in conPair:
                    newConv = deepcopy(conversation)
                    newConv['messages'][index] = item
                    allConv.append(newConv)
                    



        return  allConv     
    
    def seperate_rec_items(self, dataset):
        logger.info("Augment Conversations Seperate Recommended Items")
        augment_dataset = []
        for conv_dict in tqdm(dataset):
            if len(conv_dict['items']) > 0 :
                for movie in conv_dict['items']:
                    if not  isinstance(movie, list):
                        augment_conv_dict = deepcopy(conv_dict)
                        augment_conv_dict['items'] = movie
                        augment_dataset.append(augment_conv_dict)

        return augment_dataset

    def spec_seperate_rec_items(self, dataset):
        logger.info("Augment Conversations Seperate Recommended Items For Prediction")
        augment_dataset = []
        for conv_dict in tqdm(dataset):
            if len(conv_dict['items']) > 0 :
                for movie in conv_dict['items']:
                    if not  isinstance(movie, list):
                        augment_conv_dict = deepcopy(conv_dict)
                        augment_conv_dict['items'] = movie
                        augment_dataset.append(augment_conv_dict)
            else: 
                augment_conv_dict = deepcopy(conv_dict)           
                augment_dataset.append(augment_conv_dict)
        return augment_dataset
    
    def get_items_info_in_context_text(self, utt, text_token_ids):
        entity_ids_in_context = []   # [n_entities_in_context_text]
        entities_mask_in_context = []  # [utter_len]
        entity_mask_in_context = []
        entity_masks_in_context = [] # [n_entities_in_context_text, <=utter_len]
        for word in utt["text"]:
            entityId = self.word_is_entity(word)
            if entityId:
                entity_ids_in_context.append(entityId)
                entities_mask_in_context.append(-1)
                entity_mask_in_context.append(-1)
                entity_masks_in_context.append(copy(entity_mask_in_context))
            else:
                entities_mask_in_context.append(0)
                entity_mask_in_context.append(0)
            entity_mask_in_context[-1] = 0
        
        # padding entity_masks_in_context
        utter_len = len(text_token_ids)
        for i in range(len(entity_masks_in_context)):
            entity_masks_in_context[i] += [0] * (utter_len - len(entity_masks_in_context[i]))   
        return entity_ids_in_context, entities_mask_in_context, entity_masks_in_context
    
    def word_is_entity(self, word):
        try:
            if word.startswith('@') and word[1:].isdigit():
                ID = word[1:]
                return True
        except Exception as e:
        # Handle any exceptions (e.g., invalid input, unexpected data)
            return False
        return False
    
    def spec_augment_and_add_add_entities_mask(self, raw_conv_dict):
        logger.info("Augment Conversations")
        augmented_conv_dicts = []
        
        for conv in tqdm(raw_conv_dict):
            context_tokens, context_entities_initiator, context_entities_respondent, context_items_initiator, context_items_respondent, items_mask_in_contexts, item_masks_in_contexts, item_ids_in_contexts = [], [], [], [], [], [], [], []
            pad_utters = []
            entity_set_initiator, entity_set_respondent = set(), set()
            item_set_initiator, item_set_respondent = set(), set()
            for utt in conv['messages']: 
                text_tokens, entities, items, rec_items, items_mask_in_context, item_masks_in_context, item_ids_in_context = \
                    utt["text"], utt["entities"], utt["items"], utt["rec_items"], utt["items_mask_in_context"], utt['item_masks_in_context'], utt['item_ids_in_context']
                
                
                
                if utt['senderWorkerId'] ==  conv['initiatorWorkerId']:
                    for entity in entities:
                        if entity not in entity_set_initiator:
                            entity_set_initiator.add(entity)
                            context_entities_initiator.append(entity)

                    for item in items:
                        if item not in item_set_initiator:
                            item_set_initiator.add(item)
                            context_items_initiator.append(item)
                else :
                    for entity in entities:
                        if entity not in entity_set_respondent:
                            entity_set_respondent.add(entity)
                            context_entities_respondent.append(entity)

                    for item in items:
                        if item not in item_set_respondent:
                            item_set_respondent.add(item)
                            context_items_respondent.append(item)   

                context_tokens.append(text_tokens)  # [n_utter, utter_len]   
                items_mask_in_contexts.append(items_mask_in_context)  # [n_utter, utter_len]
                # entity_masks_in_context = [n_entities_in_utter_text, utter_len]
                padded_entity_masks_in_context = self.padd_entity_masks_in_context(pad_utters, item_masks_in_context) # [n_entities_in_utter_text, n_utter, utter_len]
                item_masks_in_contexts.extend(padded_entity_masks_in_context) # [n_entities_in_context_text, n_utter, utter_len]
                item_ids_in_contexts.extend(item_ids_in_context)  # [n_entities_in_context_text]
                        

            
                pad_utters.append([0]*len(text_tokens))
                conv_dict = {
                    "conversationId" : conv['conversationId'],
                    "current_worker_id": utt['senderWorkerId'],
                    "initiatorWorkerId": conv['initiatorWorkerId'],
                    "respondentWorkerId": conv['respondentWorkerId'],
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_entities_initiator": copy(context_entities_initiator),
                    "context_entities_respondent": copy(context_entities_respondent),
                    "context_items_initiator": copy(context_items_initiator),
                    "context_items_respondent": copy(context_items_respondent),
                    "context_items": copy(items),
                    "items_mask_in_context": copy(items_mask_in_contexts),
                    "item_masks_in_context": copy(item_masks_in_contexts),
                    "item_ids_in_context": copy(item_ids_in_contexts),
                    "items": rec_items,
                }
                augmented_conv_dicts.append(conv_dict)
                

        return augmented_conv_dicts

    def augment_and_add_add_entities_mask(self, raw_conv_dict):
        logger.info("Augment Conversations")
        augmented_conv_dicts = []
        
        for conv in tqdm(raw_conv_dict):
            context_tokens, context_entities_initiator, context_entities_respondent, context_items_initiator, context_items_respondent, items_mask_in_contexts, item_masks_in_contexts, item_ids_in_contexts = [], [], [], [], [], [], [], []
            pad_utters = []
            entity_set_initiator, entity_set_respondent = set(), set()
            item_set_initiator, item_set_respondent = set(), set()
            for utt in conv['messages']: 
                text_tokens, entities, items, rec_items, items_mask_in_context, item_masks_in_context, item_ids_in_context = \
                    utt["text"], utt["entities"], utt["items"], utt["rec_items"], utt["items_mask_in_context"], utt['item_masks_in_context'], utt['item_ids_in_context']
                
                
                if len(context_tokens) > 0:
                    conv_dict = {
                        "conversationId" : conv['conversationId'],
                        "current_worker_id": utt['senderWorkerId'],
                        "initiatorWorkerId": conv['initiatorWorkerId'],
                        "respondentWorkerId": conv['respondentWorkerId'],
                        "context_tokens": copy(context_tokens),
                        "response": text_tokens,
                        "context_entities_initiator": copy(context_entities_initiator),
                        "context_entities_respondent": copy(context_entities_respondent),
                        "context_items_initiator": copy(context_items_initiator),
                        "context_items_respondent": copy(context_items_respondent),
                        "context_items": copy(items),
                        "items_mask_in_context": copy(items_mask_in_contexts),
                        "item_masks_in_context": copy(item_masks_in_contexts),
                        "item_ids_in_context": copy(item_ids_in_contexts),
                        "items": rec_items,
                    }
                    augmented_conv_dicts.append(conv_dict)

                if utt['senderWorkerId'] ==  conv['initiatorWorkerId']:
                    for entity in entities:
                        if entity not in entity_set_initiator:
                            entity_set_initiator.add(entity)
                            context_entities_initiator.append(entity)

                    for item in items:
                        if item not in item_set_initiator:
                            item_set_initiator.add(item)
                            context_items_initiator.append(item)
                else :
                    for entity in entities:
                        if entity not in entity_set_respondent:
                            entity_set_respondent.add(entity)
                            context_entities_respondent.append(entity)

                    for item in items:
                        if item not in item_set_respondent:
                            item_set_respondent.add(item)
                            context_items_respondent.append(item)   

                context_tokens.append(text_tokens)  # [n_utter, utter_len]   
                items_mask_in_contexts.append(items_mask_in_context)  # [n_utter, utter_len]
                # entity_masks_in_context = [n_entities_in_utter_text, utter_len]
                padded_entity_masks_in_context = self.padd_entity_masks_in_context(pad_utters, item_masks_in_context) # [n_entities_in_utter_text, n_utter, utter_len]
                item_masks_in_contexts.extend(padded_entity_masks_in_context) # [n_entities_in_context_text, n_utter, utter_len]
                item_ids_in_contexts.extend(item_ids_in_context)  # [n_entities_in_context_text]
                           

                
                pad_utters.append([0]*len(text_tokens))

        return augmented_conv_dicts



    def augment_and_shifting(self, raw_conv_dict):
        logger.info("Augment Conversations shifting")
        augmented_conv_dicts = []
        
        for conv in tqdm(raw_conv_dict):
            conv_copy = deepcopy(conv)  # Create a copy of the conversation
            while conv_copy['context_tokens']:
                augmented_conv_dicts.append(deepcopy(conv_copy))  # Add the current state to the list
                conv_copy['context_tokens'].pop(0)  # Remove the first message
        
        return augmented_conv_dicts
   
    def augment_nlp_context(self, raw_conv_dict):
        logger.info("Augment NLP")
        augmented_conv_dicts = []
        # Set the seed
        
        for conv in tqdm(raw_conv_dict):
            augmented_conv_dicts.append(conv)
            # Create a copy of the conversation
            for index in range(10):
                conv_copy = deepcopy(conv)
                for i, sen in enumerate(conv_copy['context_tokens']):
                    random.seed(index * len(conv_copy['context_tokens']) + i)
                    # Calculate the length of the sentence
                    sentence_length = len(sen)
                    
                    # Determine the number of tokens to delete (20% of the sentence length)
                    num_tokens_to_delete = int(sentence_length * 0.2)
                    
                    # Generate random indices to delete
                    indices_to_delete = random.sample(range(sentence_length), num_tokens_to_delete)
                    
                    # Create a new list excluding the tokens at the indices to delete
                    augmented_sentence = [token for i, token in enumerate(sen) if i not in indices_to_delete]
                    conv_copy['context_tokens'][i] = augmented_sentence
                resp = conv_copy['response']
                # Calculate the length of the sentence
                sentence_length = len(resp)
                
                # Determine the number of tokens to delete (20% of the sentence length)
                num_tokens_to_delete = int(sentence_length * 0.2)
                
                # Generate random indices to delete
                indices_to_delete = random.sample(range(sentence_length), num_tokens_to_delete)
                
                # Create a new list excluding the tokens at the indices to delete
                augmented_resp = [token for i, token in enumerate(resp) if i not in indices_to_delete]
                conv_copy['response'] = augmented_resp  
                augmented_conv_dicts.append(conv_copy)  
        
        return augmented_conv_dicts 
    

    
    
    def padd_entity_masks_in_context(self, pad_utters, entity_masks_in_context):
        # pad_utters = [n_utter, utter_len]
        # entity_masks_in_context = [n_entities_in_utter_text, utter_len]
        padded_entity_masks_in_context = []
        for entity_mask_in_context in entity_masks_in_context:
            # entity_mask_in_context = [utter_len]
            entity_mask_in_context = pad_utters + [entity_mask_in_context] # [n_utter, utter_len]
            padded_entity_masks_in_context.append(entity_mask_in_context)

        return padded_entity_masks_in_context # [n_entities_in_utter_text, n_utter, utter_len]
    
    def get_pad_utters(self, context_tokens):
        # context_tokens = [n_utter, utter_len]
        pad_utters = []
        for utter in context_tokens:
            pad_utter = [0] * len(utter)
            pad_utters.append(pad_utter)

        return pad_utters

    
    

    def batchify(self, batch):
        batch_context_tokens = []
        batch_item_id = []
        batch_input_ids = []
        batch_target_pos = []
        batch_input_mask = []
        batch_sample_negs = []
        batch_current_worker_id = []
        batch_initiator_worker_id = []
        batch_respondent_worker_id = []
        batch_response= []
        batch_context_entities_initiator = []
        batch_context_entities_respondent = []
        batch_context_items_initiator = []
        batch_context_items_respondent = []
        batch_context_items = []
        batch_items_mask_in_context = []
        batch_item_masks_in_context = []
        batch_item_ids_in_context =[]
     

        for conv_dict in batch:
            if self.mode == 'conv':
                original_context = self._get_original_context_for_rec(conv_dict['context_tokens'])
                batch_context_tokens.append(original_context)
            else:    
                original_context = self._get_original_context_for_rec(conv_dict['context_tokens'])
                batch_context_tokens.append(original_context)
            batch_current_worker_id.append(conv_dict['current_worker_id'])
            batch_initiator_worker_id.append(conv_dict['initiatorWorkerId'])
            batch_respondent_worker_id.append(conv_dict['respondentWorkerId'])
            batch_response.append(self.build_conv_response(conv_dict['response']))
            batch_context_entities_initiator.append(conv_dict['context_entities_initiator'])
            batch_context_entities_respondent.append(conv_dict['context_entities_respondent'])
            batch_context_items_initiator.append(conv_dict['context_items_initiator'])
            batch_context_items_respondent.append(conv_dict['context_items_respondent'])
            batch_context_items.append(conv_dict['context_items'])
            batch_items_mask_in_context.extend(self.build_items_mask_in_context(conv_dict['items_mask_in_context']))
            batch_item_masks_in_context.append(self.build_sample_item_mask_in_context(conv_dict['item_masks_in_context']))
            batch_item_ids_in_context.extend(conv_dict['item_ids_in_context'])
            batch_item_id.append(conv_dict['items'])
            
            

        
        if self.mode == 'conv':
            batch_context_tokens = padded_tensor(batch_context_tokens,
                                pad_idx=self.special_token_idx['pad'],
                                pad_tail=True,
                                max_len=self.context_truncate)
        else:    
            batch_context_tokens = padded_tensor(batch_context_tokens,
                                pad_idx=self.special_token_idx['pad'],
                                pad_tail=True,
                                max_len=self.context_truncate)
        batch_response = padded_tensor(batch_response,
                                       pad_idx=self.special_token_idx['pad'],
                                       max_len=self.response_truncate,
                                       pad_tail=True)
       
        
        batch_context_entities_initiator =  padded_tensor(batch_context_entities_initiator,
                                                 pad_idx=-1,
                                                 pad_tail=True,
                                                 max_len=None)
        
        batch_context_entities_respondent =  padded_tensor(batch_context_entities_respondent,
                                                 pad_idx=-1,
                                                 pad_tail=True,
                                                 max_len=None)
        
        batch_context_items_initiator =  padded_tensor(batch_context_items_initiator,
                                                 pad_idx=-1,
                                                 pad_tail=True,
                                                 max_len=None)
        
        batch_context_items_respondent =  padded_tensor(batch_context_items_respondent,
                                                 pad_idx=-1,
                                                 pad_tail=True,
                                                 max_len=None)

                                                 
        batch_context_items =  padded_tensor(batch_context_items,
                                                pad_idx=-1,
                                                pad_tail=True,
                                                max_len=None)
        
      
        
        
        
       
        

      
        batch = {
            'worker_ids':  torch.tensor(batch_current_worker_id, dtype=torch.long ) ,
            'initiator_ids':  torch.tensor(batch_initiator_worker_id, dtype=torch.long ) ,
            'respondent_ids':  torch.tensor(batch_respondent_worker_id, dtype=torch.long ) ,
            'context': batch_context_tokens, 
            'response': batch_response,
            'context_mask': (batch_context_tokens != 0).long(),
            'context_pad_mask': (batch_context_tokens == 0).long(),
            'context_entities_initiator_ids' : batch_context_entities_initiator, 
            'context_entities_respondent_ids': batch_context_entities_respondent, 
            'context_items_initiator_ids' : batch_context_items_initiator,
            'context_items_respondent_ids' : batch_context_items_respondent,
            'context_items_ids' : batch_context_items ,
            'items_mask_in_context' : torch.tensor(batch_items_mask_in_context, dtype=torch.long ),
            'item_masks_in_context' : batch_item_masks_in_context, 
            'item_ids_in_context' : torch.tensor(batch_item_ids_in_context, dtype=torch.long ),
            'movie_to_rec': torch.tensor(batch_item_id, dtype=torch.long )
        }

      

        return batch

    def build_conv_response(self, response):
        response = add_start_end_token_idx(
            truncate(response, max_length=self.response_truncate - 2),
            start_token_idx=self.special_token_idx['start'],
            end_token_idx=self.special_token_idx['end']
        )

        return response
    def build_conv_context_tokens(self, context_tokens):
        context_tokens = [utter + [self.special_token_idx['start']] for utter in context_tokens]
        context_tokens[-1] = context_tokens[-1][:-1]
        context_tokens = truncate(merge_utt(context_tokens), max_length=self.context_truncate, truncate_tail=False)
        return context_tokens
    
    def _get_original_context_for_rec(self, context_tokens):
        # Insert special token into context. And flat the context.
        # Args:
        #     context_tokens (list of list int): 
        # Returns:
        #     compat_context (list int): 
        
        compact_context = []
        for i, utterance in enumerate(context_tokens):
            utterance = deepcopy(utterance)
            if i != 0 and self.special_token_idx['end']:
                utterance.insert(0, self.special_token_idx['end'])
            compact_context.append(utterance)
        compat_context = truncate(merge_utt(compact_context),
                                  self.context_truncate - 2,
                                  truncate_tail=False)
        compat_context = add_start_end_token_idx(compat_context,
                                                 self.special_token_idx['start'],
                                                 self.special_token_idx['end'])
        return compat_context  # List[int]
    
    def build_item_seq_for_session_rec(self, conv_dict):
        item_id = conv_dict['item']
        interaction_history = conv_dict['context_items']
        if 'interaction_history' in conv_dict:
            interaction_history = conv_dict['interaction_history'] + interaction_history

        input_ids, target_pos, input_mask, sample_negs = self._process_history(
            interaction_history, item_id)

        return input_ids, target_pos, input_mask, sample_negs
    
    def _neg_sample(self, item_set):
        item = random.randint(1, self.n_items)
        while item in item_set:
            item = random.randint(1, self.n_items)
        return item

    def _process_history(self, context_items, item_id=None):
        input_ids = truncate(context_items,
                             max_length=None,
                             truncate_tail=False)
        input_mask = [1] * len(input_ids)
        sample_negs = []
        seq_set = set(input_ids)
        for _ in input_ids:
            sample_negs.append(self._neg_sample(seq_set))

        if item_id is not None:
            target_pos = input_ids[1:] + [item_id]
            return input_ids, target_pos, input_mask, sample_negs
        else:
            return input_ids, input_mask, sample_negs
    
    
    def build_sample_item_mask_in_context(self, item_masks_in_context):
        # entity_masks_in_context = [n_eic, n_utter, utter_len]
        if not item_masks_in_context:
            return torch.tensor([])
             
        sample_item_mask_in_context = []

        for item_mask_in_context in item_masks_in_context:
            # entity_mask_in_context = [n_utter, utter_len]
            item_mask_in_context = self.build_items_mask_in_context(item_mask_in_context) # (seq_len)
            sample_item_mask_in_context.append(item_mask_in_context)
        # sample_entity_mask_in_context_len = str([len(entity_mask_in_context) for entity_mask_in_context in sample_entity_mask_in_context])
        # logger.info(f'{sample_entity_mask_in_context_len}')

        sample_item_mask_in_context = padded_tensor(
            sample_item_mask_in_context,
            pad_idx=self.special_token_idx['pad'],
            pad_tail=True,
            max_len=self.context_truncate)

        return sample_item_mask_in_context # (n_eic, seq_len)
    
    def build_items_mask_in_context(self, items_mask_in_context: List[List[int]]):
        items_mask_in_context = self._get_original_context_for_rec(items_mask_in_context) # (seq_len)
        items_mask_in_context = torch.LongTensor(items_mask_in_context)
        items_mask_in_context = (items_mask_in_context == -1).long()

        return items_mask_in_context    



if __name__ == "__main__":
    special_token_idx =  {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'pad_entity': 0,
            'pad_word': 0
        }
    DatasetSISSF(special_token_idx, 1500, 0.02, 256, 30,  "data/dataset")
