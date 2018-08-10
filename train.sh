python src/seq2seq.py \
        data/ja_dataset_train.pkl \
        data/en_dataset_train.pkl \
        data/ja_vocab.pkl \
        data/en_vocab.pkl \
        --validation_source data/ja_dataset_dev.pkl \
        --validation_target data/en_dataset_dev.pkl \
        --gpu 0 \
        --validation_interval 1 \
        --slack
