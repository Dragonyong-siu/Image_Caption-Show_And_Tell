# coco_eval

import nlgeval
from nlgeval import NLGEval
nlgeval = NLGEval()

def coco_eval(model, dataloader):
  model.eval()

  references = list()
  hypothesis = list()

  data_tqdm = tqdm(dataloader, total = len(dataloader))
  for step, (images, input_ids, targets, all_caps) in enumerate(data_tqdm):
    with torch.no_grad():
      batch = torch.stack(images).shape[0]

      with torch.cuda.amp.autocast(): 
        scores = model(torch.stack(images).to(device))
      
      references.extend(all_caps)

      samples = scores.argmax(2) 
      for i in range(batch):
        indices = torch.where(samples[i] > 3)[0]
        hypothesis.extend([' '.join([hyper_parameters['tokenizer'].decode(token) for token in samples[i][indices].long().tolist()])])
      
      assert len(references) == len(hypothesis)
  
  return references, hypothesis


valid_dataloader = torch.utils.data.DataLoader(
      valid_dataset, 
      batch_size = 8,
      num_workers = caption_config.num_workers,
      shuffle = False,
      sampler = SequentialSampler(valid_dataset),
      pin_memory = False,
      collate_fn = collate_fn)

references, hypothesis = coco_eval(gru_decoder.to(device), valid_dataloader)

nlgeval = NLGEval()
nlgeval.compute_metrics(references, hypothesis)
