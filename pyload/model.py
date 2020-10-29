# %%

from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd
import random as rn
import re, os
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Input, Embedding, Dropout, LayerNormalization, Concatenate, Lambda
from tensorflow.keras.models import Model
from sklearn.metrics import fbeta_score,precision_score,recall_score
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.backend import concatenate
from tensorflow.keras.optimizers import Adamax
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from transformers import GPT2TokenizerFast
import datetime
import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def tokenizer():
  tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
  tokenizer.add_special_tokens({"cls_token" : "<cls>","eos_token" : "<eos>","pad_token": "<pad>"})
  return tokenizer

tokenizer=tokenizer()
vocab_size=len(tokenizer)

def loss_function():
  loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
  reduction=tf.keras.losses.Reduction.NONE)
  return loss_func

def maskedLoss(y_true,y_pred):
  #shape: (batch,size,max_length) - as we are using reduction=NONE in loss_function
  loss_func=loss_function()
  loss=loss_func(y_true,y_pred)
  #mask for each sentence. Will be 0 for padded values and 1 for others 
  #shape: (batch_size,max_length)
  mask=tf.math.equal(y_true[:,:,tokenizer.pad_token_id],0)
  mask=tf.cast(mask,dtype=loss.dtype)
  loss_1=tf.math.multiply(mask,loss)
  mean_loss=tf.reduce_sum(loss_1,axis=1)/tf.reduce_sum(mask,axis=1)
  return mean_loss

def remove_padded_zeros(y_true,y_pred):
  #get word indices using argmax for all sentences in batch and flatten
  y_true_argmax=tf.reshape(tf.argmax(y_true,axis=-1),[-1])
  #remove padded zeros
  y_true_final=tf.gather(y_true_argmax,tf.where(tf.not_equal(y_true_argmax,tokenizer.pad_token_id)))
  y_true_final=tf.reshape(y_true_final,[-1])

  #get predicted word indices using argmax
  y_pred_argmax=tf.argmax(y_pred,axis=-1)
  y_pred_argmax=tf.reshape(y_pred_argmax,[-1])
  #remove padded zeros' predicted output
  y_pred_final=tf.gather(y_pred_argmax,tf.where(tf.not_equal(y_true_argmax,tokenizer.pad_token_id)))
  y_pred_final=tf.reshape(y_pred_final,[-1])
  return y_true_final,y_pred_final

def fbetaScore(y_true,y_pred):
    y_true_final,y_pred_final=remove_padded_zeros(y_true,y_pred)
    return tf.compat.v1.py_func(lambda a, b : fbeta_score(a,b,average='micro',beta=0.5), 
    (y_true_final, y_pred_final), tf.double)

def precisionScore(y_true, y_pred):
  y_true_final,y_pred_final=remove_padded_zeros(y_true,y_pred)
  return tf.compat.v1.py_func(lambda a, b : precision_score(a,b,average='micro'),
  (y_true_final, y_pred_final), tf.double)

def recallScore(y_true, y_pred):
  y_true_final,y_pred_final=remove_padded_zeros(y_true,y_pred)
  return tf.compat.v1.py_func(lambda a, b : recall_score(a,b,average='micro'),
  (y_true_final, y_pred_final), tf.double)

class gpt2_token_attention():
  def __init__(self,hidden_units,max_length,dropout_rate):
    self.hidden_units=hidden_units
    self.max_length=max_length
    self.dropout_rate=dropout_rate

    self.encoder_inputs = Input(shape=(None,),dtype=tf.int64,name="enc_input_layer")
    self.encoder_mask = Input(shape=(None,),dtype=tf.int64,name="encoder_mask")
    self.decoder_input = Input(shape=(None,),dtype=tf.int64,name='dec_input_layer')
    self.decoder_mask = Input(shape=(None,),dtype=tf.int64,name="decoder_mask")

    self.embed_layer = Embedding(vocab_size,output_dim=hidden_units,name='embed_layer')
    self.encoder_GRU_layer_1=GRU(hidden_units,dropout=dropout_rate,return_sequences=True,name='encoder_GRU_layer_1')
    self.encoder_GRU_layer_2=GRU(hidden_units,dropout=dropout_rate,return_sequences=True,name='encoder_GRU_layer_2')
    self.encoder_GRU_layer_3=GRU(hidden_units,dropout=dropout_rate,return_sequences=True,
    return_state=True,name='encoder_GRU_layer_3')

    self.weights_dense_layer1=Dense(hidden_units,name='weights_dense_layer1')
    self.weights_dense_layer2=Dense(hidden_units,name='weights_dense_layer2')
    self.score_dense_layer=Dense(1,name='score_dense_layer')

    self.decoder_GRU_layer_1=GRU(hidden_units,dropout=dropout_rate,return_sequences=True,
    return_state=True,name='decoder_GRU_layer_1')
    self.decoder_GRU_layer_2=GRU(hidden_units,dropout=dropout_rate,return_sequences=True,name='decoder_GRU_layer_2')
    self.layer_norm_layer=LayerNormalization(name='layer_norm_layer')
    self.decoder_dense_layer = Dense(vocab_size,activation='softmax',name='decoder_dense_layer')

  def create_model(self):
    #fixing numpy RS
    np.random.seed(42)
    #fixing tensorflow RS
    tf.random.set_seed(32)
    #python RS
    rn.seed(12)

    enc_embed=self.embed_layer(self.encoder_inputs)
    encoder_out=self.encoder_GRU_layer_1(enc_embed)
    encoder_out=self.encoder_GRU_layer_2(encoder_out)
    encoder_out,encoder_hidden=self.encoder_GRU_layer_3(encoder_out)

    decoder_hidden=tf.nn.tanh(encoder_hidden)
    all_outputs=[]
    for i in range(self.max_length):
      #teacher forcing - giving actual output of previous time step as input - initial input is 'START' 
      dec_inp=tf.gather(self.decoder_input,[i],axis=1)
      dec_mask=tf.gather(self.decoder_mask,[i],axis=1)

      #we are doing this to broadcast addition along the time axis to calculate the score
      decoder_hidden_with_time_axis=tf.expand_dims(decoder_hidden,1)

      #getting context_vector from attention layer
      score=self.score_dense_layer(tf.nn.tanh(self.weights_dense_layer1(decoder_hidden_with_time_axis)
      +self.weights_dense_layer2(encoder_out)))
      score=tf.squeeze(score,[-1])

      scores_mask=tf.cast(self.encoder_mask,dtype=tf.bool)
      padding_mask = tf.logical_not(scores_mask)
      score-=1.e9 * tf.cast(padding_mask, dtype=score.dtype)

      attention_weights=tf.nn.softmax(score,axis=1)
      attention_weights = tf.expand_dims(attention_weights, 1)
      context_vector=tf.matmul(attention_weights,encoder_out)
      context_vector=tf.squeeze(context_vector,1)

      context_vector*=tf.cast(tf.cast(dec_mask,dtype=tf.bool),dtype=context_vector.dtype)

      dec_inp=self.embed_layer(dec_inp)
      dec_inp=tf.concat([tf.expand_dims(context_vector,1),dec_inp],axis=-1)
      decoder_out,decoder_hidden=self.decoder_GRU_layer_1(dec_inp,initial_state=decoder_hidden)
      decoder_out=self.layer_norm_layer(decoder_out)
      decoder_out=self.decoder_GRU_layer_2(decoder_out)
      decoder_out=self.layer_norm_layer(decoder_out)
      decoder_out=self.decoder_GRU_layer_2(decoder_out)
      decoder_out=self.layer_norm_layer(decoder_out)
      out=self.decoder_dense_layer(decoder_out)
      all_outputs.append(out)

    decoder_outputs=Lambda(lambda x: concatenate(x,axis=1))(all_outputs)

    my_model=Model(inputs=[self.encoder_inputs,self.encoder_mask,self.decoder_input,self.decoder_mask],
    outputs=decoder_outputs,name='my_model')
    
    my_model.compile(optimizer=Adamax(), loss=maskedLoss,metrics=[precisionScore,recallScore,fbetaScore])
    return my_model
# %%
