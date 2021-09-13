# XLM with online word to word translations

## Additional features on top of XLM

1. Label smoothing loss (--label_sm)
2. Freezing embedding layer (--freeze_emb)
3. Online training with word to word translation as source sentence and original as target sentence

## TODO

1. Option to use label smoothing on only certain tasks
2. Integrate cross translation loss
3. Add contrastive learning used [here](https://arxiv.org/abs/2105.09501) on parallel and extend it to monolingual data 
