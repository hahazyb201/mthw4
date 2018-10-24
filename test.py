#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way. 
"""


from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
import numpy as np
import math
from torch import nn

from io import open

import matplotlib
#if you are running on the gradx/ugradx/ another cluster, 
#you will need the following line
#if you run on a local machine, you can comment it out
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
import numpy as np

logging.basicConfig(level=logging.DEBUG,
                                        format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid, 
# it can be very easy to confict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15


class Vocab:
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_sentence(vocab, sentence):
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
    # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        """Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM. 
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        "*** YOUR CODE HERE ***"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
      
        h = hidden_size
        n = input_size
        
        self.W_hi = nn.Parameter(torch.Tensor(h, h).uniform_(0, 0.001))
        self.W_hf = nn.Parameter(torch.Tensor(h, h).uniform_(0, 0.001))
        self.W_ho = nn.Parameter(torch.Tensor(h, h).uniform_(0, 0.001))
        self.W_hj = nn.Parameter(torch.Tensor(h, h).uniform_(0, 0.001))
        self.W_xi = nn.Parameter(torch.Tensor(h, n).uniform_(0, 0.001))
        self.W_xf = nn.Parameter(torch.Tensor(h, n).uniform_(0, 0.001))
        self.W_xo = nn.Parameter(torch.Tensor(h, n).uniform_(0, 0.001))
        self.W_xj = nn.Parameter(torch.Tensor(h, n).uniform_(0, 0.001))
        self.b_i = nn.Parameter(torch.Tensor(h).uniform_())
        self.b_f = nn.Parameter(torch.Tensor(h).uniform_())
        self.b_o = nn.Parameter(torch.Tensor(h).uniform_())
        self.b_j = nn.Parameter(torch.Tensor(h).uniform_())
        
        
        self.t = 0
        self.x = {}
#         self.h = {}
        self.c = {}
        self.ct = {}
        
        self.input_gate = {}
        self.forget_gate = {}
        self.output_gate = {}
        self.cell_update = {}
        
        
        
        # return output, hidden
      


    def forward(self, input, hidden):
        "*** YOUR CODE HERE ***"
        # print (self.embedding)
        self.t += 1
        t = self.t
        embedding = self.embedding(input).view(1,1,-1)
        self.input_gate[t] = nn.Sigmoid(np.dot(self.W_hi,hidden)+np.dot(self.W_xi,embedding)+self.b_i)
        self.forget_gate[t] = nn.Sigmoid((np.dot(self.W_hf,hidden)+np.dot(self.W_xf,embedding)+self.b_f))
        self.output_gate[t] = nn.Sigmoid((np.dot(self.W_ho,hidden)+np.dot(self.W_xo,embedding)+self.b_o))
        self.cell_update[t] = nn.Tanh((np.dot(self.W_hj,hidden)+np.dot(self.W_xj,embedding)+self.b_j))
        
        self.c[t] = self.input_gate[t]*self.cell_update[t]+self.forget_gate[t]*self.c[t-1]
        self.ct[t] = tanh(self.c[t])
        hidden = self.output_gate[t]*self.ct[t]
       # self.x[t] = embedding
        return self.output_gate[t], hidden
        #raise NotImplementedError
        #return output, hidden

    def get_initial_hidden_state(self):
        #return torch.zeros(1, 1, self.hidden_size, device=device)
        return nn.Parameter(torch.Tensor(self.hidden_size).uniform_())





# class EncoderRNN(nn.Module):
#   """the class for the enoder RNN
#   """
#   def __init__(self, input_size, hidden_size):
#       super(EncoderRNN, self).__init__()
# #         self.hidden_size = hidden_size
# #         """Initilize a word embedding and bi-directional LSTM encoder
# #         For this assignment, you should *NOT* use nn.LSTM. 
# #         Instead, you should implement the equations yourself.
# #         See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
# #         You should make your LSTM modular and re-use it in the Decoder.
# #         """
#       self.input_size = input_size
#       self.embedding_size = 32
#       self.hidden_size = hidden_size
#       # self.no_of_layers = no_of_layers

#       #randomly initialize
#       self.embedding = nn.Parameter(torch.Tensor(input_size, self.embedding_size).uniform_())

#       # for i in range(self.no_of_layers):
#               #parameters for forward LSTM
#       hs = self.hidden_size  # hidden state
#       es = self.embedding_size  # embedding state
        
#       #weights
# #         h0 = nn.Parameter(torch.Tensor(1, hs).uniform_())
#       self.c0 = nn.Parameter(torch.Tensor(1, hs).uniform_())
#       self.ins = hs + es
#       self.wf = nn.Parameter(torch.Tensor(self.ins, hs).uniform_(0, 0.001))
#       self.wi = nn.Parameter(torch.Tensor(self.ins, hs).uniform_(0, 0.001))
#       self.wc = nn.Parameter(torch.Tensor(self.ins, hs).uniform_(0, 0.001))
#       self.wo = nn.Parameter(torch.Tensor(self.ins, hs).uniform_(0, 0.001))

#       #bias
#       self.bf = nn.Parameter(torch.Tensor(hs).uniform_())
#       self.bi = nn.Parameter(torch.Tensor(hs).uniform_())
#       self.bc = nn.Parameter(torch.Tensor(hs).uniform_())
#       self.bo = nn.Parameter(torch.Tensor(hs).uniform_())


#       #output layer
#       self.output = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.input_size).uniform_())
#       self.bo = nn.Parameter(torch.Tensor(self.input_size).uniform_())


#   def forward(self, input, hidden):

#           #embedding layer
#           embed = self.embedding[input.data, :]

#           #seq length
#           self.seq_length = embed.size()[0]
#           self.batch_size = input.size()[1]

#           #initial inputs
#           h_forward = [embed[t,:,:] for t in range(self.seq_length)]
#           h_backward = [embed[t,:,:] for t in reversed(range(self.seq_length))]
#           # for i in range(self.no_of_layers):

#           #forward LSTM
#           h_forward = self.run_lstm(hidden, self.c0, self.wf, self.wi, self.wc, self.wo, self.bf, self.bi, self.bc, self.bo, h_forward, dropout)

#           #backward LSTM
#           h_backward = self.run_lstm(hidden, self.c0, self.wf, self.wi, self.wc, self.wo, self.bf, self.bi, self.bc, self.bo, h_backward, dropout)
            
#           out = [] 
#           for timestep in range(self.seq_length):
#                   fts = timestep - 1
#                   bts = self.seq_length - timestep - 2
#                   fh = h_forward[fts] if fts >= 0 else fh0.expand(self.batch_size, self.hidden_size)
#                   bh = h_backward[bts] if bts >= 0 else bh0.expand(self.batch_size, self.hidden_size)

#                   cat_h = torch.cat((fh, bh), 1)

#                   #output layer
#                   wout = self.output
#                   output = torch.mm(cat_h, wout) + self.bo

#                   #softmax
#                   output = torch.exp(output)
#                   output = torch.log(output / torch.sum(output, dim=1).unsqueeze(1).expand(input.size()[1], self.vocab_size))

#                   out.append(output)
                    
#           return torch.cat(out, 0), torch.cat(h_forward, h_backward)


#   def run_lstm(self, h0, c0, wf, wi, wc, wo, bf, bi, bc, bo, input):

#           hidden_units = []
#           for timestep, x in enumerate(input):
#                   hprev = h0.expand(self.batch_size, self.hidden_size) if timestep == 0 else h
#                   cprev = c0.expand(self.batch_size, self.hidden_size) if timestep == 0 else c

#                   #concatenate h(t-1) and x(t)
#                   concat_hx = torch.cat((hprev, x), 1)

#                   ft = torch.sigmoid(torch.mm(concat_hx, wf) + bf)
#                   it = torch.sigmoid(torch.mm(concat_hx, wi) + bi)
#                   ct = torch.tanh(torch.mm(concat_hx, wc) + bc)
#                   ot = torch.sigmoid(torch.mm(concat_hx, wo) + bo)

#                   c = ft * cprev + it * ct
#                   h = ot * torch.tanh(ct)
#                   hidden_units.append(h)

#           return hidden_units
#   def get_initial_hidden_state(self):
#               return torch.zeros(1, 1, self.hidden_size, device=device)
    
    
    
#     def __init__(self, input_size, hidden_size):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         """Initilize a word embedding and bi-directional LSTM encoder
#         For this assignment, you should *NOT* use nn.LSTM. 
#         Instead, you should implement the equations yourself.
#         See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
#         You should make your LSTM modular and re-use it in the Decoder.
#         """
#         "*** YOUR CODE HERE ***"
#         # for embedding, first para: rows(# words), second para: columns
                
#         self.input_size = m = input_size
#         self.hidden_size = n = hidden_size
#         self.embedding_size = 32
#         self.embedding = nn.Embedding(input_size, self.embedding_size)

#           # n = hidden unit, m = embedding dimensinality 
# #         self.U = np.random.normal(0, 0.001, (n,n))
# #         self.U_Z = np.random.normal(0, 0.001, (n,n))
# #         self.U_R = np.random.normal(0, 0.001, (n,n))
# #         self.W = np.random.normal(0, 0.001, (n,m))
# #         self.W_Z = np.random.normal(0, 0.001, (n,m))
# #         self.W_R = np.random.normal(0, 0.001, (n,m))

# #         self.W_lr = ...
# #         self.W_rl = ...
# #         self.W_o = ...
# #         self.b_lr = ...
# #         self.b_rl = ...
# #         self.b_o = ...

# #         self.W_f = ...
# #         self.W_b = ...
# #         self.U_f = ...
# #         self.U_b = ...
# #         self.V = ... (?)
                
# #         self.b_1 = ...
# #         self.b_2 = ...
# #         self.b_3 = ...
            
#         self.softmax = nn.LogSoftmax()
#         self.init_params()
                
                

#     def forward(self, input, hidden):
#         """runs the forward pass of the encoder
#         returns the output and the hidden state
#         """
#         "*** YOUR CODE HERE ***"
                
#         # view(1,1,-1): -1 means put all the left element to this dimension
# #         embedded = self.embedding(input).view(1, 1, -1)
                
                
# #         seq_len = len(input)
#         embedding = self.embedding(input).view(1, 1, -1)
                
#         output = torch.tensor(torch.zeros(seq_len, self.input_size), requires_grad=False)
#         h_lr = torch.tensor(torch.rand(seq_len + 1, self.hidden_size), requires_grad=False)
#         h_rl = torch.tensor(torch.rand(seq_len + 1, self.hidden_size), requires_grad=False)
#         h_lr[0,:] = hidden
                
                
#         for t in range(self.hidden_size):
# #             word = embedding[:,:,t]
# #             v = embedded[word,:]
#               embedding = self.embedding(input)
# #             concat = torch.cat([word, hidden], 1)
# #             hidden = concat.matmul(self.W_lr)
# #             hidden += self.b_lr
# #             hidden = torch.tanh(hidden)
#             h_i = torch.tanh(self.W.matmul(embedding).matmul(input)+self.U.matmul(r_i*h_lr[t]))
#             z_i = 
#             r_i = 
#             h_lr[t+1,:] = hidden
                
#         h_rl[seq_len,:] = hidden


# #         embedding = self.embedding[input,:]  ???
# #         seq_len = len(input)
        
                
    
                
                        
#         return output, hidden

        

        
class AttnLayer(nn.Module):
    def __init__(self,hidden_size):
        super(AttnLayer, self).__init__()
        self.hidden_size = hidden_size
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.v.data.normal_(mean=0,std =(1.0/math.sqrt(self.v.size(0))) )
        self.attn = nn.Linear(self.hidden_size*2,hidden_size)
            
    def forward(self,hidden,encoder_outputs,max_length):
        encoder_outputs = encoder_outputs.transpose(0,1)
        hidden = hidden.repeat(max_length,1,1).transpose(0,1)
        eng = F.tanh(self.attn(torch.cat([hidden,encoder_outputs],2)))
        eng = eng.transpose(2,1)
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1)
        eng = torch.bmm(v,eng)
        attn_eng = ent.squeeze(1)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
                
        "*** YOUR CODE HERE ***"
        self.embedded_size = hidden_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = AttnLayer(hidden_size)
        self.LSTM = EncoderRNN(hidden_size,hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        "*** YOUR CODE HERE ***"
        embedding = self.embedding(input).view(1,1-1)
        embedding_d = self.dropout(embedding)
        attn_weights = self.attn(hidden[-1],encoder_outputs,self.max_length)
        cont = attn_weights.bmm(encoder_outputs.transpose(0,1)).transpose(0,1)
        biInput = torch.cat((embedding_d,cont),2)
        output,hidden = self.LSTM(biInput,hidden) 
        log_softmax = F.log_softmax(self.out(output.squeeze(0)))
        return log_softmax, hidden, attn_weights

    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.get_initial_hidden_state()
    encoder.train()
    decoder.train()

    "*** YOUR CODE HERE ***"
    optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_hidden_left = encoder.get_initial_hidden_state()
    encoder_hidden_right = encoder.get_initial_hidden_state()
    encoder_outputs_left = torch.zeros(max_length, encoder.hidden_size)
    encoder_outputs_right = torch.zeros(max_length, encoder.hidden_size)
    encoder_outputs = torch.zeros(max_length, 2*encoder.hidden_size)
    loss = 0
    for i in range(input_length):
        encoder_output_left, encoder_hidden_left = encoder.forward(input_tensor[i], encoder_hidden_left)
        encoder_output_right, encoder_hidden_right = encoder.forward(input_tensor[input_length-i-1], encoder_hidden_right)
        encoder_outputs_left[i] = encoder_output_left[0,0]
        encoder_outputs_right[input_length-i-1] = encoder_output_right[0,0]

    for i in range(input_length):
        encoder_outputs[i]=torch.cat((encoder_outputs_left[i],encoder_outputs_right[i]))

        
    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_hidden_right

    for i in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[i])
        decoder_input = target_tensor[i] 


    loss.backward()

    optimizer.step()
    return loss.item() 




######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_index]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions):
    "*** YOUR CODE HERE ***"
    pic=plt.figure()
    axes=pic.add_subplot(111)
    axes.matshow(attentions,cmap=plt.cm.bone)
    axes.set_xticklabels(['']+input_sentence.split(' ')+['<EOS>'],rotation=90)
    axes.set_yticklabels(['']+output_words)
    axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()




def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)


def clean(strx):
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=5000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    while iter_num < args.n_iters:
        iter_num += 1
        training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs)) #SGD  
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,    #numerical tensor
                     decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab)


if __name__ == '__main__':
    main()