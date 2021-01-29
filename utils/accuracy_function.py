def accuracy(scores, targets, k):
  batch = targets.size(0)
  _, indices = scores.topk(k, 1, True, True)
  correct = indices.eq(targets.to(device).view(-1, 1).expand_as(indices))
  correct_total = correct.view(-1).float().sum() 
  return correct_total.item() * (100 / batch)
