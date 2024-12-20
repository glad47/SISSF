## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14

import argparse
import datetime
import hashlib
import os
import shutil
import socket
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gc
import torch
import torch.nn as nn
import torch.optim

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler
from sklearn.model_selection import KFold
from model.conversation_history_encoder import ConversationHistoryEncoder
from util.util import enumerateWithEstimate
from data.data_processing import DatasetSISSF
from util.logconf import logging
from model.semantic_fusion import SemanticFusion
from model.recommendation_task import RecommenderModule
from model.conversational_task import ConversationalModule

import time

import logging as lg


from util.logFile import log
from util.logconf import logging
logger = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

# Used for computeClassificationLoss and logMetrics to index into metrics_t/metrics_a
# METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
# METRICS_FN_LOSS_NDX = 2
# METRICS_ALL_LOSS_NDX = 3

# METRICS_PTP_NDX = 4
# METRICS_PFN_NDX = 5
# METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_R_1_NDX = 4
METRICS_R_10_NDX = 5
METRICS_R_50_NDX = 6
METRICS_NDCG_1_NDX = 10
METRICS_HIT_1_NDX = 11
METRICS_MRR_1_NDX = 12

METRICS_NDCG_10_NDX = 13
METRICS_HIT_10_NDX = 14
METRICS_MRR_10_NDX = 15


METRICS_DIST_2_NDX = 16
METRICS_DIST_3_NDX = 17
METRICS_DIST_4_NDX = 18



METRICS_SIZE = 20

class SISSF:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=300,
            type=int,
        )
        parser.add_argument('--test-batch-size',
            help='Test Batch size to use for training',
            default=300,
            type=int,
        )

        parser.add_argument('--valid-batch-size',
            help='Validation Batch size to use for training',
            default=300,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=1,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=500,
            type=int,
        )
        parser.add_argument('--kfold',
            help='Number of fold to split the dataset',
            default=3,
            type=int,
        )
        parser.add_argument('--mode',
            help="semantic -> semantic fusion, rec -> recommendation model, conv -> conversational model",
            default="semantic",
        )
        parser.add_argument('--test-mode',
            help="Test mode",
            default=False,
        ) 
        parser.add_argument('--lr',
            help="The learning rate",
            default=1e-5,
            type=float
        )
        parser.add_argument('--lr-bert',
            help="A separate learning rate for BERT components",
            default=1e-5,
            type=float
        )
        parser.add_argument('--gradient-clip',
            help="The maximum norm of the gradients for gradient clipping",
            default=0.1,
            type=float,
        )

        parser.add_argument('--factor',
            help="The factor by which the learning rate will be reduced",
            default=0.5,
            type=float,
        )
        parser.add_argument('--early-stop',
            help="Enables early stopping to prevent overfitting",
            default=True,
        )
        
        parser.add_argument('--impatience',
            help='The number of epochs with no improvement after which training will be stopped',
            default=25,
            type=int,
        )
        # '/content/drive/MyDrive/SegmentationProject/content/drive/MyDrive/project/seg_2024-02-09_10.04.25_reduceFP.300000.state'
        parser.add_argument('--finetune',
            help="Start finetuning from this model.",
            default=None,
        )

        parser.add_argument('--resume-learning',
            help="Start finetuning from this model.",
            default=None,
        )

        parser.add_argument('--dataset',
            help=" Dataset",
            default='data/dataset',
        )

        parser.add_argument('--dataset-type',
            help="Dataset Type ReDial -> ReDial datatset, INSPIRED -> INSPIRED dataset",
            default='ReDial',
        )

        parser.add_argument('--pretrain-token-embeddings-weights',
            help=" pretrin embeddings weight",
            default=None,
        )


        parser.add_argument('--semantic-fusion-weights',
            help="pertrained weights for semantic fusion",
            default=None,
        )

        parser.add_argument('--context-truncate',
            help="The maximum length of the context to consider is 1024 tokens.",
            default=256,
            type=int,
        )
        parser.add_argument('--response-truncate',
            help="The maximum length of the response to generate is 30 tokens.",
            default=30,
            type=int,
        )
        parser.add_argument('--token-freq-th',
            help="Token Frequency",
            default=100,
            type=int,
        )
        parser.add_argument('--weight-th',
            help="weight-th",
            default=0.3,
            type=float,
        )
        parser.add_argument('--scale',
            help="This could be a scaling factor for the input data",
            default=1.0,
            type=float,
        )
        parser.add_argument('--embeddings-scale',
            help="This embeddings scale indicator",
            default=True,
        )
        parser.add_argument('--token-emb-dim',
            help="The dimensionality of the token embeddings",
            default=300,
            type=int,
        )
        parser.add_argument('--kg-emb-dim',
            help="The dimensionality of the knowledge graph embeddings",
            default=768,
            type=int,
        )
        parser.add_argument('--social-emb-dim',
            help="The dimensionality of the social graph embeddings",
            default=768,
            type=int,
        )
        parser.add_argument('--interaction-emb-dim',
            help="The dimensionality of the user-item graph embeddings",
            default=768,
            type=int,
        )

        parser.add_argument('--user-emb-dim',
            help="The dimensionality of the user embeddings",
            default=768,
            type=int,
        )
        parser.add_argument('--num-bases',
            help="The number of bases",
            default=8,
            type=int,
        )
        parser.add_argument('--n-heads',
            help="The number of head",
            default=2,
            type=int,
        )
        parser.add_argument('--n-layers',
            help="The number of layers",
            default=2,
            type=int,
        )
        parser.add_argument('--ffn-size',
            help="The size of the feedforward network",
            default=300,
            type=int,
        )

        parser.add_argument('--rec-ffn-size',
            help="The size of the feedforward network",
            default=300,
            type=int,
        )

        parser.add_argument('--conv-ffn-size',
            help="The size of the feedforward network",
            default=300,
            type=int,
        )
        parser.add_argument('--dropout',
            help="The dropout rate",
            default=0.1,
            type=float,
        )

        parser.add_argument('--rec-dropout',
            help="The dropout rate",
            default=0.1,
            type=float,
        )


        parser.add_argument('--conv-dropout',
            help="The dropout rate",
            default=0.1,
            type=float,
        )

        parser.add_argument('--sem-dropout',
            help="The dropout rate",
            default=0.1,
            type=float,
        )
        parser.add_argument('--attention-dropout',
            help="The dropout rate for the attention weights",
            default=0.0,
            type=float,
        )
        parser.add_argument('--relu-dropout',
            help="The dropout rate after the ReLU activation",
            default=0.1,
            type=float,
        )
        parser.add_argument('--learn-positional-embeddings',
            help="The positional embeddings indicator",
            default=False,
        )

        parser.add_argument('--learn-positional-embeddings-decoder',
            help="The positional embeddings indicator",
            default=True,
        )
        parser.add_argument('--n-positions',
            help="The maximum number of positions for positional embeddings",
            default=1024,
            type=int,
        )
        
        parser.add_argument('--tem',
            help="softmax temperature",
            default=0.07,
            type=float,
        )
        

        parser.add_argument('--rec-layers',
            help="number of layers to recommender transformer",
            default=8,
            type=int,
        )



        parser.add_argument('--rec-heads',
            help="number of head to recommender transformer",
            default=4,
            type=int,
        )
        
        
        parser.add_argument('--conv-heads',
            help="number of head to recommender transformer",
            default=2,
            type=int,
        )


        parser.add_argument('--conv-layers',
            help="number of layers to recommender transformer",
            default=2,
            type=int,
        )

        parser.add_argument('--cumulative-prob-th',
            help="top-p sampling (also known as nucleus sampling), cumulative probability threshold",
            default=0.4,
            type=float,
        )


        parser.add_argument('--max-norm',
            help="gerd norm",
            default=0.5,
            type=float,
        )
        
        parser.add_argument('--weight-decay',
            help="weight-decay",
            default=0,
            type=float,
        )
        parser.add_argument('--reduction',
            help="The reduction indicator",
            default=False,
        )
        
        parser.add_argument('--tb-prefix',
            default='projectSemantic',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('--comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='Semantic Fusion',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0
        self.start_epoch = 1
        self.trn_writer = None
        self.val_writer = None
        self.test_writer = None
        self.special_token_idx =  {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'pad_entity': 0,
            'pad_word': 0
        }
        self.val_score = 0
        self.best_score = 0
        self.endure_count = 0
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.kf = KFold(n_splits=self.cli_args.kfold)
        torch.manual_seed(42)
        self.initDataset()
        self.sample_percent = 0.7 # 70% of the data
        self.sampler = self.get_sampler()
        self.initModel()

        if self.cli_args.finetune:
            checkpoint = torch.load(self.cli_args.finetune, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state'])
            log.info("Fine Tuning {}, {}".format(type(self).__name__, self.cli_args))


        if self.cli_args.resume_learning:  
            if self.cli_args.mode == 'semantic':
                checkpoint = torch.load(self.cli_args.resume_learning, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state'])
                torch.set_rng_state(checkpoint['rng_state'])
                self.totalTrainingSamples_count = checkpoint['totalTrainingSamples_count']
                self.best_score = checkpoint['best_score']
                self.endure_count = checkpoint['endure_count']
                self.start_epoch = checkpoint['epoch'] + 1
            elif self.cli_args.mode == 'rec':
                checkpoint = torch.load(self.cli_args.resume_learning, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state'])
                self.semantic_model.load_state_dict(checkpoint['semantic_model_state'])
                torch.set_rng_state(checkpoint['rng_state'])
                self.totalTrainingSamples_count = checkpoint['totalTrainingSamples_count']
                self.best_score = checkpoint['best_score']
                self.endure_count = checkpoint['endure_count']
                self.start_epoch = checkpoint['epoch'] + 1
            elif self.cli_args.mode == 'conv':
                checkpoint = torch.load(self.cli_args.resume_learning, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state'])
                self.semantic_model.load_state_dict(checkpoint['semantic_model_state'])
                torch.set_rng_state(checkpoint['rng_state'])
                self.totalTrainingSamples_count = checkpoint['totalTrainingSamples_count']
                self.best_score = checkpoint['best_score']
                self.endure_count = checkpoint['endure_count']
                self.start_epoch = checkpoint['epoch'] + 1       
           

        if self.use_cuda:
            if self.cli_args.mode == 'semantic':
                log.info("Using CUDA; {} devices. {}".format(torch.cuda.device_count(), self.device))
                self.model = self.model.to(self.device)
                if torch.cuda.device_count() > 1:
                    self.model = nn.DataParallel(self.model) 
            elif self.cli_args.mode == 'rec':
                log.info("Using CUDA; {} devices. {}".format(torch.cuda.device_count(), self.device))
                self.model = self.model.to(self.device)
                self.semantic_model = self.semantic_model.to(self.device)
                if torch.cuda.device_count() > 1:
                    self.model = nn.DataParallel(self.model) 
                    self.semantic_model = nn.DataParallel(self.semantic_model)
            elif self.cli_args.mode == 'conv':
                log.info("Using CUDA; {} devices. {}".format(torch.cuda.device_count(), self.device))
                self.model = self.model.to(self.device)
                self.semantic_model = self.semantic_model.to(self.device)
                if torch.cuda.device_count() > 1:
                    self.model = nn.DataParallel(self.model) 
                    self.semantic_model = nn.DataParallel(self.semantic_model)        

    
        self.initOptimizer()

        if self.cli_args.resume_learning: 
            checkpoint = torch.load(self.cli_args.resume_learning, map_location='cpu')
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            batch_size =self.cli_args.batch_size
            lr =self.cli_args.lr
            ts =self.cli_args.test_mode
            epochs =self.cli_args.epochs
            self.cli_args = checkpoint['sys_argv']
            self.cli_args.batch_size = batch_size
            self.cli_args.lr = lr
            self.cli_args.test_mode = ts
            self.cli_args.epochs = epochs
            self.cli_args.pretrain_token_embeddings_weights = None
            log.info("Resume learning {}, {}".format(type(self).__name__, self.cli_args))
            current_lr = next(iter(self.optimizer.param_groups))['lr']
            log.info(f"Resuming with learning rate: {current_lr}")
            # Check if the current learning rate is different from the new learning rate
            if current_lr != self.cli_args.lr:
                # Update the learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.cli_args.lr
                log.info(f"Updated learning rate to: {self.cli_args.lr}")
            
      
            
        

    def initDataset(self):
        self.dataset = DatasetSISSF(self.special_token_idx,self.cli_args.token_freq_th, self.cli_args.weight_th, 
                               self.cli_args.context_truncate, self.cli_args.response_truncate, self.cli_args.dataset, self.cli_args.dataset_type, self.cli_args.mode)
        
    

    def seed_worker(self,w_id):
        worker_seed = torch.initial_seed() % 2**32
        torch.manual_seed(worker_seed)

    
    
    def get_sampler(self):
        dataset_size = len(self.dataset.train_data)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        split = int(np.floor(self.sample_percent * dataset_size))
        sample_indices = indices[:split]
        return SubsetRandomSampler(sample_indices)

    def initDataloaders(self):
        self.sampler = self.get_sampler()
        batch_size = self.cli_args.batch_size
        valid_batch_size = self.cli_args.valid_batch_size
        test_batch_size = self.cli_args.test_batch_size
        
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
            valid_batch_size *= torch.cuda.device_count()
            test_batch_size *= torch.cuda.device_count()
            
        
        self.train_dl = DataLoader(
            self.dataset.train_data,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            sampler=self.sampler,
            collate_fn= self.dataset.batchify
        )

        self.val_dl = DataLoader(
            self.dataset.valid_data,
            batch_size=valid_batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            collate_fn= self.dataset.batchify
        )
        
      
            
        self.test_dl = DataLoader(
            self.dataset.test_data,
            batch_size=test_batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            collate_fn= self.dataset.batchify
        )    

      
    

    def initTestDataloaders(self):
        test_batch_size = self.cli_args.test_batch_size
        if self.use_cuda :
            test_batch_size *= torch.cuda.device_count()
        self.test_dl = DataLoader(
            self.dataset.test_data,
            batch_size=test_batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            collate_fn= self.dataset.batchify
        )

           
    def initModel(self):
        if self.cli_args.mode == 'semantic':
            self.model =  SemanticFusion(self.cli_args.dataset, self.cli_args.dataset_type, self.cli_args.social_emb_dim, self.cli_args.interaction_emb_dim, self.cli_args.user_emb_dim,
                                        self.dataset.users,self.dataset.items,
                                        self.cli_args.kg_emb_dim,
                                        0, self.cli_args.num_bases, self.cli_args.dataset, self.cli_args.pretrain_token_embeddings_weights, self.cli_args.token_emb_dim,
                                        self.cli_args.n_heads, self.cli_args.n_layers,self.cli_args.ffn_size, len(self.dataset.tok2ind),self.dataset.tok2ind,
                                        self.cli_args.dropout, self.cli_args.attention_dropout, self.cli_args.relu_dropout,
                                        self.special_token_idx['pad'],self.special_token_idx['start'],self.cli_args.learn_positional_embeddings,
                                        self.cli_args.embeddings_scale, self.cli_args.reduction, self.cli_args.n_positions,
                                        self.cli_args.tem, self.cli_args.sem_dropout, self.device )
        elif self.cli_args.mode == 'rec': 
            if self.cli_args.semantic_fusion_weights:
                self.cli_args.pretrain_token_embeddings_weights = None
            
            self.semantic_model = SemanticFusion(self.cli_args.dataset, self.cli_args.dataset_type, self.cli_args.social_emb_dim, self.cli_args.interaction_emb_dim, self.cli_args.user_emb_dim,
                                        self.dataset.users,self.dataset.items,
                                        self.cli_args.kg_emb_dim,
                                        0, self.cli_args.num_bases, self.cli_args.dataset, self.cli_args.pretrain_token_embeddings_weights, self.cli_args.token_emb_dim,
                                        self.cli_args.n_heads, self.cli_args.n_layers,self.cli_args.ffn_size, len(self.dataset.tok2ind),self.dataset.tok2ind,
                                        self.cli_args.dropout, self.cli_args.attention_dropout, self.cli_args.relu_dropout,
                                        self.special_token_idx['pad'],self.special_token_idx['start'],self.cli_args.learn_positional_embeddings,
                                        self.cli_args.embeddings_scale, self.cli_args.reduction, self.cli_args.n_positions,
                                        self.cli_args.tem, self.cli_args.sem_dropout, self.device )
            
            if self.cli_args.semantic_fusion_weights and self.cli_args.resume_learning is None :
                checkpoint = torch.load(self.cli_args.semantic_fusion_weights, map_location='cpu')
                self.semantic_model.load_state_dict(checkpoint['model_state'])
             


            self.model = RecommenderModule(self.cli_args.user_emb_dim,self.cli_args.interaction_emb_dim, self.cli_args.social_emb_dim,
                                           self.cli_args.kg_emb_dim, self.dataset.n_items, 0,
                                           self.dataset.n_users,self.cli_args.rec_heads,self.cli_args.rec_ffn_size,
                                           self.cli_args.rec_dropout, self.cli_args.attention_dropout,self.cli_args.rec_layers, self.device )  
        elif self.cli_args.mode == 'conv':
            
            if self.cli_args.semantic_fusion_weights:
                self.cli_args.pretrain_token_embeddings_weights = None
            
            self.semantic_model = SemanticFusion(self.cli_args.dataset,self.cli_args.dataset_type, self.cli_args.social_emb_dim, self.cli_args.interaction_emb_dim, self.cli_args.user_emb_dim,
                                        self.dataset.users,self.dataset.items,
                                        self.cli_args.kg_emb_dim,
                                        0, self.cli_args.num_bases, self.cli_args.dataset, self.cli_args.pretrain_token_embeddings_weights, self.cli_args.token_emb_dim,
                                        self.cli_args.n_heads, self.cli_args.n_layers,self.cli_args.ffn_size, len(self.dataset.tok2ind),self.dataset.tok2ind,
                                        self.cli_args.dropout, self.cli_args.attention_dropout, self.cli_args.relu_dropout,
                                        self.special_token_idx['pad'],self.special_token_idx['start'],self.cli_args.learn_positional_embeddings,
                                        self.cli_args.embeddings_scale, self.cli_args.reduction, self.cli_args.n_positions,
                                        self.cli_args.tem, self.cli_args.sem_dropout, self.device )
            
            if self.cli_args.semantic_fusion_weights and self.cli_args.resume_learning is None :
                checkpoint = torch.load(self.cli_args.semantic_fusion_weights, map_location='cpu')
                self.semantic_model.load_state_dict(checkpoint['semantic_model_state'])
            
            state_dict = self.semantic_model.state_dict()

            # Access the embedding weights using the key for the embedding layer
            embedding_weights = state_dict['conversation_encoder.embedding.weight']
            self.model = ConversationalModule(self.cli_args.dataset,self.cli_args.dataset_type, self.cli_args.user_emb_dim, 
                                             self.cli_args.social_emb_dim, self.cli_args.conv_heads,
                                             self.cli_args.conv_ffn_size,self.cli_args.conv_dropout, self.cli_args.conv_dropout,
                                             self.cli_args.conv_dropout, self.cli_args.conv_layers, len(self.dataset.tok2ind), 
                                             self.special_token_idx['start'], self.special_token_idx['pad'],self.special_token_idx['end'], 
                                             self.cli_args.response_truncate, self.dataset.decoder_token_prob_weight,embedding_weights, 
                                             self.cli_args.token_emb_dim,self.cli_args.embeddings_scale, 
                                             self.cli_args.learn_positional_embeddings_decoder,self.cli_args.n_positions, self.cli_args.cumulative_prob_th ,self.device )
        log.info("complete init model")

    def initOptimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.cli_args.lr, weight_decay=self.cli_args.weight_decay)
        # return SGD(self.segmentation_model.parameters(), lr=0.001, momentum=0.99)

        log.info("complete init optimizer")


    def initTensorboardWriters(self):
        msg= 'sf'
        if self.cli_args.mode == 'semantic' :
            msg= 'sf'
        elif self.cli_args.mode == 'rec' :  
            msg= 'rec' 
        elif self.cli_args.mode == 'conv' :  
            msg= 'conv'     
        
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '_trn_'+msg+'_' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '_val_'+msg+'_' + self.cli_args.comment)
            self.test_writer = SummaryWriter(
                log_dir=log_dir + '_trn_'+msg+'_' + self.cli_args.comment)

    

    def doTraining(self, epoch_ndx,fold_num):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(self.train_dl.dataset), device=self.device)
        self.model.train()
        
        batch_iter = enumerateWithEstimate(
            self.train_dl,
            "E{} Training, {} fold".format(epoch_ndx, fold_num),
            start_ndx=self.train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            # loss= self.model(batch_tup)
            logger.info(f"********trn******{batch_ndx}******{fold_num}********")
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, self.train_dl.batch_size, trnMetrics_g)
            loss_var.backward()
           
            self.optimizer.step()
       
            

            

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

         # Memory management
        del batch_iter
        gc.collect()                           # Collect garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, fold_num):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(self.val_dl.dataset), device=self.device)
            self.model.eval()
            if self.cli_args.mode == 'rec' or  self.cli_args.mode == 'conv':
                self.semantic_model.eval()

            batch_iter = enumerateWithEstimate(
                self.val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=self.val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                logger.info(f"********val******{batch_ndx}*****{fold_num}*********")
                self.computeBatchLoss(batch_ndx, batch_tup, self.val_dl.batch_size, valMetrics_g)
             
               
               
            del batch_iter
            gc.collect()                           # Collect garbage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()   

        return valMetrics_g.to('cpu')
    

    def doTesting(self, epoch_ndx):
        with torch.no_grad():
            testMetrics_g = torch.zeros(METRICS_SIZE, len(self.test_dl.dataset), device=self.device)
            self.model.eval()
            if self.cli_args.mode == 'rec' or  self.cli_args.mode == 'conv':
                self.semantic_model.eval()

            batch_iter = enumerateWithEstimate(
                self.test_dl,
                "E{} Testing".format(epoch_ndx),
                start_ndx=self.test_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                logger.info(f"********test******{batch_ndx}**************")
                self.computeBatchLoss(batch_ndx, batch_tup, self.test_dl.batch_size, testMetrics_g)
            
            del batch_iter
            gc.collect()                           # Collect garbage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()     

        return testMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        if self.cli_args.mode == 'semantic':
            loss, TP, FP, FN = self.model(batch_tup)
            start_ndx = batch_ndx * batch_size
            end_ndx = start_ndx + batch_tup['worker_ids'].shape[0]

            with torch.no_grad():
                metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss
                metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = TP
                metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = FP
                metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = FN
                
            return loss
        elif self.cli_args.mode == 'rec':
            rec_loss, R_at_1, R_at_10, R_at_50 = self.model(self.semantic_model, batch_tup)
            start_ndx = batch_ndx * batch_size
            end_ndx = start_ndx + batch_tup['worker_ids'].shape[0]
            with torch.no_grad():
                metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = rec_loss
                metrics_g[METRICS_R_1_NDX, start_ndx:end_ndx] = R_at_1
                metrics_g[METRICS_R_10_NDX, start_ndx:end_ndx] = R_at_10
                metrics_g[METRICS_R_50_NDX, start_ndx:end_ndx] = R_at_50
                
               
            return rec_loss    
        elif self.cli_args.mode == 'conv':
            conv_loss, Dist_2, Dist_3, Dist_4 = self.model(self.semantic_model, batch_tup)
            start_ndx = batch_ndx * batch_size
            end_ndx = start_ndx + batch_tup['worker_ids'].shape[0]
            with torch.no_grad():
                metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = conv_loss
                metrics_g[METRICS_DIST_2_NDX, start_ndx:end_ndx] = Dist_2
                metrics_g[METRICS_DIST_3_NDX, start_ndx:end_ndx] = Dist_3
                metrics_g[METRICS_DIST_4_NDX, start_ndx:end_ndx] = Dist_4
                
               
            return conv_loss       
    def calMetrics(self, metrics_t, fold_num, epochMetrics_g):

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()
        metrics_dict = {}
        if self.cli_args.mode == 'semantic':
            

            allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

            
            metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

            metrics_dict['percent_all/tp'] = \
                sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
            metrics_dict['percent_all/fn'] = \
                sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
            metrics_dict['percent_all/fp'] = \
                sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100


            precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
                / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
            recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
                / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

            metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
                / ((precision + recall) or 1)
            epochMetrics_g[METRICS_LOSS_NDX, fold_num] = torch.tensor(metrics_dict['loss/all'], dtype=torch.float32, device='cuda')
            epochMetrics_g[METRICS_TP_NDX, fold_num] = torch.tensor(sum_a[METRICS_TP_NDX], dtype=torch.float32, device='cuda') 
            epochMetrics_g[METRICS_FP_NDX, fold_num] = torch.tensor(sum_a[METRICS_FP_NDX], dtype=torch.float32, device='cuda') 
            epochMetrics_g[METRICS_FN_NDX, fold_num] = torch.tensor(sum_a[METRICS_FN_NDX], dtype=torch.float32, device='cuda')
           
        elif self.cli_args.mode == 'rec':   
            
            metrics_a = metrics_t.detach().numpy()
            sum_a = metrics_a.sum(axis=1)
            assert np.isfinite(metrics_a).all()

            

            metrics_dict = {}
            metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()
         
            metrics_dict['R@1'] = metrics_a[METRICS_R_1_NDX].mean()
            metrics_dict['R@10'] = metrics_a[METRICS_R_10_NDX].mean()
            metrics_dict['R@50'] = metrics_a[METRICS_R_50_NDX].mean()
            

            epochMetrics_g[METRICS_LOSS_NDX, fold_num] = torch.tensor(metrics_dict['loss/all'], dtype=torch.float32, device='cuda')
            epochMetrics_g[METRICS_R_1_NDX, fold_num] = torch.tensor(metrics_dict['R@1'], dtype=torch.float32, device='cuda')
            epochMetrics_g[METRICS_R_10_NDX, fold_num] = torch.tensor(metrics_dict['R@10'], dtype=torch.float32, device='cuda')
            epochMetrics_g[METRICS_R_50_NDX, fold_num] = torch.tensor(metrics_dict['R@50'], dtype=torch.float32, device='cuda')


        elif self.cli_args.mode == 'conv':   
            
            metrics_a = metrics_t.detach().numpy()
            sum_a = metrics_a.sum(axis=1)
            assert np.isfinite(metrics_a).all()

            

            metrics_dict = {}
            metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()
         
            metrics_dict['DIST_2'] = metrics_a[METRICS_DIST_2_NDX].mean()
            metrics_dict['DIST_3'] = metrics_a[METRICS_DIST_3_NDX].mean()
            metrics_dict['DIST_4'] = metrics_a[METRICS_DIST_4_NDX].mean()
            

            epochMetrics_g[METRICS_LOSS_NDX, fold_num] = torch.tensor(metrics_dict['loss/all'], dtype=torch.float32, device='cuda')
            epochMetrics_g[METRICS_DIST_2_NDX, fold_num] = torch.tensor(metrics_dict['DIST_2'], dtype=torch.float32, device='cuda')
            epochMetrics_g[METRICS_DIST_3_NDX, fold_num] = torch.tensor(metrics_dict['DIST_3'], dtype=torch.float32, device='cuda')
            epochMetrics_g[METRICS_DIST_4_NDX, fold_num] = torch.tensor(metrics_dict['DIST_4'], dtype=torch.float32, device='cuda')    


            

           

        
      
       
        
       

    

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        log.info("E{} {}".format(
                epoch_ndx,
                type(self).__name__,
            ))
        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()
        metrics_dict = {}
        if self.cli_args.mode == 'semantic':
            

            allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

            
            metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

            metrics_dict['percent_all/tp'] = \
                sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
            metrics_dict['percent_all/fn'] = \
                sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
            metrics_dict['percent_all/fp'] = \
                sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100


            precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
                / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
            recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
                / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

            metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
                / ((precision + recall) or 1)

            log.info(("E{} {:8} "
                    + "{loss/all:.4f} loss, "
                    + "{pr/precision:.4f} precision, "
                    + "{pr/recall:.4f} recall, "
                    + "{pr/f1_score:.4f} f1 score"
                    ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            ))
            log.info(("E{} {:8} "
                    + "{loss/all:.4f} loss, "
                    + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
            ).format(
                epoch_ndx,
                mode_str + '_all',
                **metrics_dict,
            ))
        elif self.cli_args.mode == 'rec':   
            log.info("E{} {}".format(
                epoch_ndx,
                type(self).__name__,
            ))
            metrics_a = metrics_t.detach().numpy()
            sum_a = metrics_a.sum(axis=1)
            assert np.isfinite(metrics_a).all()

            

            metrics_dict = {}
            metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()
            metrics_dict['R@1'] = metrics_a[METRICS_R_1_NDX].mean()
            metrics_dict['R@10'] = metrics_a[METRICS_R_10_NDX].mean()
            metrics_dict['R@50'] = metrics_a[METRICS_R_50_NDX].mean()
            

           


            

            log.info(("E{} {:8} "
                    + "{loss/all:.4f} loss, "
                    + "{R@1:.4f} R@1, "
                    + "{R@10:.4f} R@10, "
                    + "{R@50:.4f} R@50"
                    ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            ))

        elif self.cli_args.mode == 'conv':   
            log.info("E{} {}".format(
                epoch_ndx,
                type(self).__name__,
            ))
            metrics_a = metrics_t.detach().numpy()
            sum_a = metrics_a.sum(axis=1)
            assert np.isfinite(metrics_a).all()

            

            metrics_dict = {}
            metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()
            metrics_dict['DIST_2'] = metrics_a[METRICS_DIST_2_NDX].mean()
            metrics_dict['DIST_3'] = metrics_a[METRICS_DIST_3_NDX].mean()
            metrics_dict['DIST_4'] = metrics_a[METRICS_DIST_4_NDX].mean()
            

           


            

            log.info(("E{} {:8} "
                    + "{loss/all:.4f} loss, "
                    + "{DIST_2:.4f} DIST_2, "
                    + "{DIST_3:.4f} DIST_3, "
                    + "{DIST_4:.4f} DIST_4"
                    ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            ))
           

        self.initTensorboardWriters()
        
        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')
        msg= 'sf'
        if self.cli_args.mode == 'semantic' :
            msg= 'sf'
        elif self.cli_args.mode == 'rec' :  
            msg= 'rec' 
        elif self.cli_args.mode == 'conv' :  
            msg= 'conv' 

        for key, value in metrics_dict.items():
            writer.add_scalar(msg + key, value, self.totalTrainingSamples_count)

        writer.flush()

      
        if self.cli_args.mode == 'semantic':
            return metrics_dict['loss/all']
        elif self.cli_args.mode == 'rec':
            return metrics_dict['R@1'], metrics_dict['R@50']
        elif self.cli_args.mode == 'conv':
            return metrics_dict['loss/all']    

    

    
    def saveModel(self, epoch_ndx, isBest=False):
        model_folder = 'output'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder, mode=0o755, exist_ok=True)

        if not os.path.exists(os.path.join(model_folder, self.cli_args.mode)):
            os.makedirs(os.path.join(model_folder, self.cli_args.mode), mode=0o755, exist_ok=True)    
              

        # Main model state file path
        file_path = os.path.join(model_folder,self.cli_args.mode, f'model_{epoch_ndx}.{self.time_str}.state')
        model = self.model

        
        
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        
        if self.cli_args.mode == 'rec':
            semantic_model = self.semantic_model     

        
            if isinstance(semantic_model, torch.nn.DataParallel):
                semantic_model = semantic_model.module

        if self.cli_args.mode == 'conv':
            semantic_model = self.semantic_model     

    
            if isinstance(semantic_model, torch.nn.DataParallel):
                semantic_model = semantic_model.module     

        
        if self.cli_args.mode == 'semantic':
            main_state = {
                'sys_argv': self.cli_args,
                'time': str(datetime.datetime.now()),
                'model_state': model.state_dict(),
                'model_name': type(model).__name__,
                'optimizer_state' : self.optimizer.state_dict(),
                'optimizer_name': type(self.optimizer).__name__,
                'epoch': epoch_ndx,
                'rng_state': torch.get_rng_state(),
                'totalTrainingSamples_count': self.totalTrainingSamples_count,
                'best_score': self.best_score,
                'endure_count': self.endure_count,
            }
        elif self.cli_args.mode == 'rec':  
            main_state = {
                'sys_argv': self.cli_args,
                'time': str(datetime.datetime.now()),
                'model_state': model.state_dict(),
                'semantic_model_state': semantic_model.state_dict(),
                'model_name': type(model).__name__,
                'optimizer_state' : self.optimizer.state_dict(),
                'optimizer_name': type(self.optimizer).__name__,
                'epoch': epoch_ndx,
                'rng_state': torch.get_rng_state(),
                'totalTrainingSamples_count': self.totalTrainingSamples_count,
                'best_score': self.best_score,
                'endure_count': self.endure_count,
            }  


        elif self.cli_args.mode == 'conv':  
            main_state = {
                'sys_argv': self.cli_args,
                'time': str(datetime.datetime.now()),
                'model_state': model.state_dict(),
                'semantic_model_state': semantic_model.state_dict(),
                'model_name': type(model).__name__,
                'optimizer_state' : self.optimizer.state_dict(),
                'optimizer_name': type(self.optimizer).__name__,
                'epoch': epoch_ndx,
                'rng_state': torch.get_rng_state(),
                'totalTrainingSamples_count': self.totalTrainingSamples_count,
                'best_score': self.best_score,
                'endure_count': self.endure_count,
            }      

        torch.save(main_state, file_path)
        
        log.info("Saved model params to {}".format(file_path))
        
        if isBest:
            best_path = os.path.join(model_folder,self.cli_args.mode, f'model_best.state')
            
            
                
            
            torch.save(main_state, best_path)
            

            log.info(f"Saved best model params to {best_path}")
           
            for file_path in [best_path]:
                with open(file_path, 'rb') as f:
                    sha1_hash = hashlib.sha1(f.read()).hexdigest()
                    log.info(f"SHA1: {sha1_hash} for {file_path}")  

        for file_path in [file_path]:
            with open(file_path, 'rb') as f:
                sha1_hash = hashlib.sha1(f.read()).hexdigest()
                log.info(f"SHA1: {sha1_hash} for {file_path}")


    

    def performTesting(self):
        # load the best model 
        best_path = os.path.join("output",self.cli_args.mode, f'model_best.state')
        checkpoint = torch.load(best_path, map_location='cpu')
        if self.cli_args.mode == 'semantic':
            self.model.load_state_dict(checkpoint['model_state'])
        elif self.cli_args.mode == 'rec': 
            self.model.load_state_dict(checkpoint['model_state'])
            self.semantic_model.load_state_dict(checkpoint['semantic_model_state']) 

        elif self.cli_args.mode == 'conv': 
            self.model.load_state_dict(checkpoint['model_state'])
            self.semantic_model.load_state_dict(checkpoint['semantic_model_state'])       
        
        # perform testing and print the result
        testMetrics_t = self.doTesting()
        expected_loss = self.logMetrics(0, 'test', testMetrics_t)
        log.info("Testing expected_loss: %.4f" % (expected_loss))

   



    


    def mainV1(self):
        log.info("Starting {}, {}, Mode {}".format(type(self).__name__, self.cli_args, self.cli_args.mode))
        self.best_score = 9999.0
        
        
        if not self.cli_args.test_mode :
            for epoch in range(self.start_epoch, self.cli_args.epochs + 1):
                self.initDataloaders()
                log.info("Epoch {} of {}".format(
                        epoch,
                        self.cli_args.epochs
                    ))
                start_time = time.time()
                trainData= self.doTraining(epoch, 0)
                self.logMetrics(epoch, 'trn', trainData)
                valMetrics_t = self.doValidation(epoch, 0)
                score = self.logMetrics(0, 'val', valMetrics_t)   
                if self.cli_args.mode == 'semantic':
                    if  score < self.best_score:
                        self.best_score = score
                        self.endure_count = 0
                        self.saveModel(epoch, True)
                    else:
                        self.endure_count += 1
                        self.saveModel(epoch, False)
                    log.info("loss: %.4f" % (score))
                elif self.cli_args.mode == 'conv':
                    if epoch % 50 == 0:
                        if  score < self.best_score:
                            self.best_score = score
                            self.endure_count = 0
                            self.saveModel(epoch, True)
                        else:
                            self.endure_count += 1
                            self.saveModel(epoch, False)
                    log.info("loss: %.4f" % (score))    

                end_time = time.time()  
                epoch_duration = end_time - start_time 
                log.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds.")

                if self.endure_count > self.cli_args.impatience :
                    log.info(f"Epoch {epoch}: early stopping.")
                    break
        
        else:
            self.initTestDataloaders()
            testMetrics_t = self.doTesting(0)
            score = self.logMetrics(0, 'test', testMetrics_t)     

    def initDataloadersV2(self, train_index, test_index):
        
        batch_size = self.cli_args.batch_size
        valid_batch_size = self.cli_args.valid_batch_size
        test_batch_size = self.cli_args.test_batch_size
        
        
            
        train_set, val_set = random_split(self.dataset.train_data, [len(train_index), len(test_index)])
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
            valid_batch_size *= torch.cuda.device_count()
            test_batch_size *= torch.cuda.device_count()
            
        
        self.train_dl = DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            collate_fn= self.dataset.batchify
        )

      

        self.val_dl = DataLoader(
            val_set,
            batch_size=valid_batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            collate_fn= self.dataset.batchify
        )


        self.test_dl = DataLoader(
            self.dataset.test_data,
            batch_size=test_batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            collate_fn= self.dataset.batchify
        )    

                        
    def mainV2(self):
        log.info("Starting {}, {}, Mode {}".format(type(self).__name__, self.cli_args, self.cli_args.mode))
        
        
        if not self.cli_args.test_mode :
            for epoch in range(self.start_epoch, self.cli_args.epochs + 1):
                log.info("Epoch {} of {}".format(
                        epoch,
                        self.cli_args.epochs
                    ))
                start_time = time.time()
                
                trainMetrics_g = torch.zeros(METRICS_SIZE, self.cli_args.kfold, device=self.device)
                valMetrics_g = torch.zeros(METRICS_SIZE, self.cli_args.kfold, device=self.device)
                for fold_index, (train_index, test_index) in enumerate(self.kf.split(self.dataset.train_data)):
                    self.initDataloadersV2(train_index, test_index)
                    trnMetrics_t = self.doTraining(epoch, fold_index)
                    self.calMetrics(trnMetrics_t, fold_index,trainMetrics_g)
                
                    valMetrics_t = self.doValidation(epoch, fold_index)
                    self.calMetrics(valMetrics_t,fold_index, valMetrics_g)
                
                
                trainData =trainMetrics_g.to('cpu')
                valData =valMetrics_g.to('cpu')
                self.logMetrics(epoch, 'trn', trainData)
                score_R_1, score_R_50 = self.logMetrics(epoch, 'val', valData)       
                 

                if self.cli_args.dataset_type == 'ReDial':   
                    score = score_R_1
                    if score > self.val_score:
                        testMetrics_t = self.doTesting(0)
                        test_score_R_1, test_score_R_50 = self.logMetrics(0, 'test', testMetrics_t) 
                        if  test_score_R_1 > self.best_score:
                                self.best_score = test_score_R_1
                                self.endure_count = 0
                                self.saveModel(epoch, True)

                else : 
                    score = score_R_1 +  score_R_50 
                    if score > self.val_score:
                        testMetrics_t = self.doTesting(0)
                        test_score_R_1, test_score_R_50 = self.logMetrics(0, 'test', testMetrics_t) 
                        if  test_score_R_1 + test_score_R_50 > self.best_score:
                                self.best_score = test_score_R_1 + test_score_R_50
                                self.endure_count = 0
                                self.saveModel(epoch, True)  
                
                        

                end_time = time.time()  
                epoch_duration = end_time - start_time 
                log.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds.")

                if self.endure_count > self.cli_args.impatience  and False :
                    log.info(f"Epoch {epoch}: early stopping.")
                    break
        
        else:
            self.initTestDataloaders()
            testMetrics_t = self.doTesting(0)
            score = self.logMetrics(0, 'test', testMetrics_t)       
          
      
    

    def main(self):
        if self.cli_args.mode == 'rec':
            self.mainV2()
        else:
            self.mainV1() 



if __name__ == '__main__':
    SISSF().main()
