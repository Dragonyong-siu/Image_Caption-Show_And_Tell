# hyper_parameters

hyper_parameters = {}

hyper_parameters['img_size'] = 224
hyper_parameters['max_len'] = 20
hyper_parameters['shrink_factor'] = 0.8
hyper_parameters['topk'] = 5
hyper_parameters['patience'] = 4
hyper_parameters['tokenizer'] = nltk_tokenizer
if hyper_parameters['tokenizer'] == bert_tokenizer:
  hyper_parameters['vocab_dim'] = 30525
elif hyper_parameters['tokenizer'] == gpt2_tokenizer:
  hyper_parameters['vocab_dim'] = 50260
else:
  hyper_parameters['vocab_dim'] = nltk_tokenizer.__len__()

hyper_parameters['grad_clip'] = 5.0
