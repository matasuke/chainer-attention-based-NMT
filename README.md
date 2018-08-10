# Chainer Simple NMT
This is just an example of simple neural machine translation implemented using Chainer.

## Requirement
```
pip install -r requirements.txt
```

## Dataset preparation
You need four files to train

1. Source language sentence file
2. Source language vocabulary file
3. Target language sentence file
4. Target language sentence file

Dataset that just contains sentences can be converted into formatted one by src/common/preprocess_tokens.py.

if you want to use validation dataset.
additional files below are necessary.

5. Source language validation sentence file
6. Target language validation sentence file

## Training with small datast

In this repository, small_parallel_enja is used.
Firstly clone it.

```
git clone https://github.com/odashi/small_parallel_enjw
```

then process it by preprocess_data.sh.

```
sh ./preprocess_data.sh
```

This script makes six files explained in Datset preparation section.
Now you have 6 files.

1. ja_dataset_train.pkl
2. ja_vocab.pkl
3. en_dataset_train.pkl
4. en_vocab.pkl
5. ja_dataset_dev.pkl
6. en_dataset_dev.pkl

# Usage
You can start training by train.sh

```
sh ./train.sh
```

