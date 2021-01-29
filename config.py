# caption_config

class caption_config:
  num_workers = 16
  batch_size = 32
  n_epochs = 50
  lr = 4e-4
  folder = 'trained_model'
  verbose = True
  verbose_step = 1
  step_scheduler = False
  validation_scheduler = True
  SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
  scheduler_params = dict(
      mode = 'min',
      factor = 0.5,
      patience = 1,
      verbose = False, 
      threshold = 0.0001,
      threshold_mode = 'abs',
      cooldown = 0, 
      min_lr = 1e-8,
      eps = 1e-08)
