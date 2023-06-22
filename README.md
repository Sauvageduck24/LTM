# LTM
Experimental Long-term memory code

This repository is an experimental implementation of the Long-term Memory paper.

Code based in https://github.com/dhlee347/pytorchic-bert

Download necesary data
-------------------------------------

[GLUE Benchmark Datasets] https://github.com/nyu-mll/GLUE-baselines
[BERT-Base, Uncased] https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

Simple Transformer (67.0 M Parameters)
-------------------------------------

```
Iter (loss=0.493): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 115/115 [07:49<00:00,  4.08s/it]
Epoch 1/3 : Average Loss 0.669
Iter (loss=0.684): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 115/115 [08:25<00:00,  4.39s/it] 
Epoch 2/3 : Average Loss 0.633
Iter (loss=0.695): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 115/115 [08:18<00:00,  4.34s/it] 
Epoch 3/3 : Average Loss 0.628
```

```
Iter(acc=0.417): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:05<00:00,  2.48it/s]
Accuracy: 0.6838235259056091
```

Simple Transformer (65.2 M Parameters)
-------------------------------------
```
Iter (loss=0.666): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 115/115 [06:48<00:00,  3.55s/it]
Epoch 1/3 : Average Loss 0.654
Iter (loss=0.639): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 115/115 [08:23<00:00,  4.38s/it] 
Epoch 2/3 : Average Loss 0.624
Iter (loss=0.485): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 115/115 [10:56<00:00,  5.71s/it] 
Epoch 3/3 : Average Loss 0.600
```

```
Iter(acc=0.708): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:05<00:00,  2.38it/s]
Accuracy: 0.6936274766921997
```

Commands to run
-------------------------------------

###Train

```
export GLUE_DIR=/path/to/glue
export SAVE_DIR=/path/to/save
export BERT_PRETRAIN=/path/to/pretrain

python classify.py \
    --task mrpc \
    --mode train \
    --train_cfg config/train_mrpc.json \
    --model_cfg config/bert_base.json \
    --data_file $GLUE_DIR/MRPC/train.tsv \
    --pretrain_file None \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir $SAVE_DIR \
    --max_len 128
```

###Evaluation

```
export GLUE_DIR=/path/to/glue
export SAVE_DIR=/path/to/save
export BERT_PRETRAIN=/path/to/pretrain

python classify.py \
    --task mrpc \
    --mode eval \
    --train_cfg config/train_mrpc.json \
    --model_cfg config/bert_base.json \
    --data_file $GLUE_DIR/MRPC/dev.tsv \
    --model_file $SAVE_DIR/model_steps_345.pt \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --max_len 128
```
