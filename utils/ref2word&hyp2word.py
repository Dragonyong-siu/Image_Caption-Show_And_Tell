def __ref2word__(all_refs):
  all_words = []
  for refs in all_refs:
    words = []
    for ref in refs:
      words.append(ref.split())
    all_words.append(words)
  return all_words

def __hyp2word__(hyps):
  words = []
  for hyp in hyps:
    words.append(hyp.split())
  return words
