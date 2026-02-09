---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: class
    dtype: string
  - name: reactants
    dtype: string
  - name: product
    dtype: string
  splits:
  - name: train
    num_bytes: 16286068
    num_examples: 40008
  - name: val
    num_bytes: 2033225
    num_examples: 5001
  - name: test
    num_bytes: 2030183
    num_examples: 5007
  download_size: 9311808
  dataset_size: 20349476
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
  - split: test
    path: data/test-*
---
