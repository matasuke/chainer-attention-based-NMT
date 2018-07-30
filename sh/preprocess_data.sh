python preprocess_tokens.py small_parallel_enja/train.ja data/ja_dataset.pickle --lang jp --tokenize --val_in_path small_parallel_enja/dev.ja --val_out_path data/ja_dataset_dev.pickle
python preprocess_tokens.py small_parallel_enja/train.en data/en_dataset.pickle --lang en --tokenize --val_in_path small_parallel_enja/dev.en --val_out_path data/en_dataset_dev.pickle
