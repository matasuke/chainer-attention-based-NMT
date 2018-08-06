python src/common/preprocess_tokens.py small_parallel_enja/train.ja \
                            data/ja_dataset_train.pkl \
                            data/ja_vocab.pkl \
                            --lang ja \
                            --normalize \
                            --cutoff 5

python src/common/preprocess_tokens.py small_parallel_enja/train.en \
                            data/en_dataset_train.pkl \
                            data/en_vocab.pkl \
                            --lang en \
                            --normalize \
                            --cutoff 5

python src/common/preprocess_tokens.py small_parallel_enja/dev.ja \
                            data/ja_dataset_dev.pkl \
                            data/ja_vocab.pkl \
                            --validation \
                            --lang ja \
                            --normalize \
                            --cutoff 5

python src/common/preprocess_tokens.py small_parallel_enja/dev.en \
                            data/en_dataset_dev.pkl \
                            data/en_vocab.pkl \
                            --validation \
                            --lang ja \
                            --normalize \
                            --cutoff 5
