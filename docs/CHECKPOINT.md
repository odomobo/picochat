  File "/mnt/c/Users/Josh/Projects/AI/picochat/scripts/base_train.py", line 195, in <module>
    optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, tied_weights_lr=tied_weights_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
  File "/mnt/c/Users/Josh/Projects/AI/picochat/nanochat/gpt.py", line 287, in setup_optimizers
    adam_parameter_count += len(adam_group['params'])
TypeError: object of type 'generator' has no len()

Hmm, but, I should be getting 'params' from a dict, and the value should be a list... what's going on?