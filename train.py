# CapBenchTrain

class CapBenchTrain(nn.Module):
  def __init__(self, model):
    super(CapBenchTrain, self).__init__()
    self.model = model
    self.loss_fn = nn.CrossEntropyLoss()

  def forward(self, images, input_ids, target):
    preds_out = self.model(images, input_ids)
    loss = self.loss_fn(preds_out.reshape(-1, hyper_parameters['vocab_dim']), target.reshape(-1))

    return loss, preds_out

net = CapBenchTrain(gru_decoder).to(device)
net.model.__encoder__.requires_grad_ = False









# caption_params 

caption_params = []
mode = 'gru'
if mode == 'gru':
  modules = [net.model.__word2embed_layer__, 
             net.model.__img2embed_layer__,
             net.model.__language_gru__,
             net.model.__fc_layer__] 

elif mode == 'gpt2':
  modules = []
           
for module in modules:
  caption_params += list(module.parameters())

print(len(caption_params))









# caption_fitter : automatic_mixed_precision

import warnings
import os
import time
from datetime import datetime
from glob import glob
from nltk.translate.bleu_score import corpus_bleu

warnings.filterwarnings('ignore')
scaler = torch.cuda.amp.GradScaler() 

class caption_fitter:
  def __init__(self, model, config, params):
    self.model = model
    self.config = config
    
    self.epoch = 0
    self.base_dir = f'/content/gdrive/My Drive/coco_image_caption/train/{self.config.folder}'
    if not os.path.exists(self.base_dir):
      os.makedirs(self.base_dir)
        
    self.log_path = f'{self.base_dir}/log.txt'
    self.params = params

    self.best_summary_loss = 10 ** 5
    self.best_bleu4 = 0.0
    self.epochs_from_improvement = 0

    self.optimizer = torch.optim.AdamW(self.params, lr = config.lr)
    self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
    self.log(f'caption_fitter prepared. device is {device}')
  
  def fit(self, train_dataloader, valid_dataloader):
    for e in range(self.config.n_epochs):
      if self.epochs_from_improvement == 20:
        break
      if self.epochs_from_improvement > 0 and self.epochs_from_improvement % hyper_parameters['patience'] == 0:
        adjust_lr(self.optimizer, hyper_parameters['shrink_factor'])

      if self.config.verbose:
        lr = self.optimizer.param_groups[0]['lr']
        timestamp = datetime.utcnow().isoformat()
        self.log(f'\n{timestamp}\nLR: {lr}')

      t = time.time()
      summary_loss, topk_accuracy = self.train_function(train_dataloader)

      self.log(f'[RESULT]: train. epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, topk_accuracy: {topk_accuracy.avg: .5f}, time: {(time.time() - t):.5f}')
      self.save(f'{self.base_dir}/last-checkpoint.bin')

      t = time.time()
      summary_loss, topk_accuracy, bleu4 = self.valid_function(valid_dataloader)

      self.log(f'[RESULT]: valid. epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, topk_accuracy: {topk_accuracy.avg: .5f}, bleu4: {bleu4:.5f}, time: {(time.time() - t):.5f}')
      if summary_loss.avg < self.best_summary_loss:
        self.best_summary_loss = summary_loss.avg
        self.model.eval()
        self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')

        for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
          os.remove(path)
      
      if bleu4 >= self.best_bleu4:
        self.best_bleu4 = bleu4
        self.epochs_from_improvement = 0  
      else:
        self.epochs_from_improvement += 1

      if self.config.validation_scheduler:
        self.scheduler.step(metrics = summary_loss.avg)

      self.epoch += 1

  def valid_function(self, valid_dataloader):
    self.model.eval()

    summary_loss = averagemeter()
    topk_accuracy = averagemeter()
    
    t = time.time()
    valid_book = tqdm(valid_dataloader, total = len(valid_dataloader))

    references = list()
    hypothesis = list()
    for step, (images, input_ids, targets, all_caps) in enumerate(valid_book):
      with torch.no_grad():
        batch = torch.stack(images).shape[0]
        
        with torch.cuda.amp.autocast(): 
          loss, scores = self.model(torch.stack(images).to(device), 
                                    torch.stack(input_ids).to(device), 
                                    torch.stack(targets).to(device)) 

        summary_loss.update(loss.detach().item(), batch)

        topk = accuracy(scores = scores.reshape(-1, hyper_parameters['vocab_dim']), 
                        targets = torch.stack(targets).reshape(-1), 
                        k = hyper_parameters['topk'])
        topk_accuracy.update(topk, batch * hyper_parameters['max_len'])

        references.extend(all_caps)

        samples = scores.argmax(2) # (N, max_len)
        for i in range(batch):
          indices = torch.where(samples[i] > 3)[0]
          hypothesis.extend([' '.join([hyper_parameters['tokenizer'].decode(token) for token in samples[i][indices].long().tolist()])])
        
        assert len(references) == len(hypothesis)
    
    bleu4 = round(corpus_bleu(__ref2word__(references), __hyp2word__(hypothesis)), 4)

    return summary_loss, topk_accuracy, bleu4

  def train_function(self, train_dataloader):
    self.model.train()

    summary_loss = averagemeter()
    topk_accuracy = averagemeter()

    t = time.time()
    train_book = tqdm(train_dataloader, total = len(train_dataloader))
    for step, (images, input_ids, targets) in enumerate(train_book):
      batch = torch.stack(images).shape[0]
     
      self.optimizer.zero_grad()

      torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, 
                                            self.params), hyper_parameters['grad_clip']) 
      
      with torch.cuda.amp.autocast(): 
        loss, scores = self.model(torch.stack(images).to(device), 
                                  torch.stack(input_ids).to(device),
                                  torch.stack(targets).to(device))  

      scaler.scale(loss).backward()         
      scaler.step(self.optimizer) 
      scaler.update()

      summary_loss.update(loss.detach().item(), batch)

      topk = accuracy(scores = scores.reshape(-1, hyper_parameters['vocab_dim']), 
                      targets = torch.stack(targets).reshape(-1), 
                      k = hyper_parameters['topk'])
      topk_accuracy.update(topk, batch * hyper_parameters['max_len'])

      if self.config.step_scheduler:
          self.scheduler.step()

      ##for p in self.params: 
      ##  print(p.grad.norm()) 

    return summary_loss, topk_accuracy
    
  def save(self, path):
    self.model.eval()
    torch.save({
        'model_state_dict': self.model.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
        'best_summary_loss': self.best_summary_loss,
        'epoch': self.epoch,
    }, path)

  def load(self, path):
    checkpoint = torch.load(path)
    self.model.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    self.best_summary_loss = checkpoint['best_summary_loss']
    self.epoch = checkpoint['epoch'] + 1
      
  def log(self, message):
    if self.config.verbose:
      print(message)
    with open(self.log_path, 'a+') as logger:
      logger.write(f'{message}\n')
      
      
      
      
  
  
  
  # caption_runner

from torch.utils.data.sampler import SequentialSampler, RandomSampler

def caption_runner():
  net.to(device)

  train_dataloader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size = caption_config.batch_size,
      sampler = RandomSampler(train_dataset),
      pin_memory = False,
      drop_last = True,
      num_workers = caption_config.num_workers,
      collate_fn = collate_fn)
  
  valid_dataloader = torch.utils.data.DataLoader(
      valid_dataset, 
      batch_size = caption_config.batch_size,
      num_workers = caption_config.num_workers,
      shuffle = False,
      sampler = SequentialSampler(valid_dataset),
      pin_memory = False,
      collate_fn = collate_fn)

  fitter = caption_fitter(model = net, config = caption_config, params = caption_params)
  fitter.fit(train_dataloader, valid_dataloader)
