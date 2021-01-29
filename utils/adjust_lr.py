def adjust_lr(optimizer, shrink_factor):
  for param_group in optimizer.param_groups:
    param_group['lr'] = param_group['lr'] * shrink_factor
  print(' adjust_LR is completed')
