# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import model

def load_best_model(hidden_units,max_length,dropout_rate,best_weights_path):
    gpt2_token_attention=model.gpt2_token_attention(hidden_units,max_length,dropout_rate)
    best_model=gpt2_token_attention.create_model()
    best_weights=tf.train.latest_checkpoint(best_weights_path)
    best_model.load_weights(best_weights)
    
    return best_model

class inference_models():
    def __init__(self,hidden_units,max_length,dropout_rate,best_weights_path):
        
        #loading model with best weights
        print("Creating model, compiling and loading best weights")
        self.best_model=load_best_model(hidden_units,max_length,dropout_rate,best_weights_path)
        print("Best weights loaded")

        #inputs for cv model
        self.dec_hidden = Input(shape=(None,))
        self.de_mask = Input(shape=(None,),dtype=tf.int64)
        self.enc_mask = Input(shape=(None,),dtype=tf.int64)
        self.enc_out = Input(shape=(None,None))

        self.score_dense_layer=self.best_model.get_layer(name='score_dense_layer')
        self.weights_dense_layer1=self.best_model.get_layer(name='weights_dense_layer1')
        self.weights_dense_layer2=self.best_model.get_layer(name='weights_dense_layer2')

        #inputs for decoder model
        self.dec_input = Input(shape=(None,))
        self.dec_hid = Input(shape=(None,))
        self.context_vec = Input(shape=(None,))

        self.embed_layer=self.best_model.get_layer(name='embed_layer')
        self.decoder_GRU_layer_1=self.best_model.get_layer(name='decoder_GRU_layer_1')
        self.decoder_GRU_layer_2=self.best_model.get_layer(name='decoder_GRU_layer_2')
        self.layer_norm_layer=self.best_model.get_layer(name='layer_norm_layer')
        self.decoder_dense_layer=self.best_model.get_layer(name='decoder_dense_layer')

    def create_inference_models(self):
        #encoder model for inference
        encoder_inputs,encoder_mask=self.best_model.inputs[:2]
        encoder_out, encoder_hidden=self.best_model.get_layer(name='encoder_GRU_layer_3').output
        inf_enc_model=Model(inputs=[encoder_inputs,encoder_mask],outputs=[encoder_out,encoder_hidden],
        name='inf_enc_model')

        #cv model for inference to get context_vector
        dec_hid_with_time_axis=tf.expand_dims(self.dec_hidden,1)
        score_1=self.score_dense_layer(tf.nn.tanh(self.weights_dense_layer1(dec_hid_with_time_axis)
        +self.weights_dense_layer2(self.enc_out)))
        score_1=tf.squeeze(score_1,[-1])
        scores_mask_1=tf.cast(self.enc_mask,dtype=tf.bool)
        padding_mask_1 = tf.logical_not(scores_mask_1)
        score_1-=1.e9 * tf.cast(padding_mask_1, dtype=score_1.dtype)
        attention_weights_1=tf.nn.softmax(score_1,axis=1)
        context_vector_1=tf.matmul(attention_weights_1,self.enc_out)
        context_vector_1=tf.squeeze(context_vector_1,1)
        context_vector_1*=tf.cast(self.de_mask,dtype=context_vector_1.dtype)
        cv_model = Model(inputs=[self.dec_hidden,self.de_mask,self.enc_mask,self.enc_out],
        outputs=context_vector_1,name='cv_model')

        #decoder model for inference
        dec_inp1=self.embed_layer(self.dec_input)
        dec_inp1=tf.concat([tf.expand_dims(self.context_vec,1),dec_inp1],axis=-1)
        dec_out,dec_hid1=self.decoder_GRU_layer_1(dec_inp1,initial_state=self.dec_hid)
        dec_out=self.layer_norm_layer(dec_out)
        dec_out=self.decoder_GRU_layer_2(dec_out)
        dec_out=self.layer_norm_layer(dec_out)
        dec_out=self.decoder_GRU_layer_2(dec_out)
        dec_out=self.layer_norm_layer(dec_out)
        out_1=self.decoder_dense_layer(dec_out)

        inf_dec_model = Model(inputs=[self.dec_input,self.context_vec,self.dec_hid],
        outputs=[out_1,dec_hid1],name='inf_dec_model')

        return inf_enc_model,cv_model,inf_dec_model

def get_inference_models(hidden_units,max_length,dropout_rate,best_weights_path):
    inf_models=inference_models(hidden_units,max_length,dropout_rate,best_weights_path)
    inf_enc_model,cv_model,inf_dec_model=inf_models.create_inference_models()

    return inf_enc_model,cv_model,inf_dec_model

class inference():
    def __init__(self,incorrect_sentence,hidden_units,max_length,dropout_rate,best_weights_path,
    tokenizer,inf_enc_model,cv_model,inf_dec_model):
        self.incorrect_sentence=incorrect_sentence
        self.max_length=max_length
        self.tokenizer=tokenizer
        self.inf_enc_model=inf_enc_model
        self.cv_model=cv_model
        self.inf_dec_model=inf_dec_model

    def predict(self):
        test_inputs = self.tokenizer.encode_plus(self.tokenizer.cls_token+self.incorrect_sentence, 
        add_special_tokens=True, max_length=self.max_length, pad_to_max_length=True,
        return_attention_mask=True, truncation=True,return_tensors='tf')
        enc_inp=test_inputs['input_ids']
        enc_mask=test_inputs['attention_mask']
        incorrect_sent=self.tokenizer.decode(enc_inp[0].numpy(),skip_special_tokens=True)

        encoder_out,encoder_hidden=self.inf_enc_model.predict([enc_inp,enc_mask])
        dec_inp=tf.expand_dims([self.tokenizer.cls_token_id], 0)
        dec_mask=tf.expand_dims([1],0)

        decoder_hidden=tf.nn.tanh(encoder_hidden)

        all_outputs1=[]
        for i in range(self.max_length):
            context_vector=self.cv_model.predict([decoder_hidden,dec_mask,enc_mask,encoder_out])
            out,decoder_hidden=self.inf_dec_model.predict([dec_inp,context_vector,decoder_hidden])

            output_index=np.argmax(out)
            all_outputs1.append(output_index)

            if output_index == self.tokenizer.eos_token_id:
                break
            
            dec_inp=tf.expand_dims([output_index], 0)
            if output_index == self.tokenizer.pad_token_id:
                dec_mask=tf.expand_dims([0],0)
            else:
                dec_mask=tf.expand_dims([1],0)

        predicted_sentence = self.tokenizer.decode(all_outputs1,skip_special_tokens=True)

        return incorrect_sent,predicted_sentence
# %%
