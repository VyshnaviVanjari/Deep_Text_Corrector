# %%

import pandas as pd
import model
import inference

hidden_units = 512
max_length = 51
dropout_rate = 0.1
best_weights_path='gpt2_token_attn_best'
tokenizer=model.tokenizer
inf_enc_model,cv_model,inf_dec_model=inference.get_inference_models(hidden_units,
max_length,dropout_rate,best_weights_path)

print("\nInference start")

test=pd.read_csv('test.csv')
for i in range(10):
  incorrect_sentence=test.iloc[i][1]
  correct_ids=tokenizer.encode(test.iloc[i][0])
  correct_sent=tokenizer.decode(correct_ids)

  inference_obj=inference.inference(incorrect_sentence,hidden_units,
  max_length,dropout_rate,best_weights_path,tokenizer,inf_enc_model,
  cv_model,inf_dec_model)

  #calling predict function
  incorrect_sent,predicted_sent=inference_obj.predict()
  print("\nIncorrect sentence: ",incorrect_sent)
  print("Correct sentence: ",correct_sent)
  print("Predicted sentence: ",predicted_sent)
  print("--"*20)
# %%
