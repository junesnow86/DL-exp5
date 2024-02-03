import csv
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'zh'

# Define special symbols and indices
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<pad>', '<unk>', '<bos>', '<eos>']

MAX_LEN = 100

def create_data_iter(file_path: str) -> Iterator[Tuple[str, str]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
            yield tuple(row)

# helper function to yield list of tokens
def yield_tokens(token_transform: Dict[str, Any], data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# function to truncate token list
def truncate_transform(token_ids: torch.Tensor, max_len: int = MAX_LEN):
    if len(token_ids) > max_len:
        return torch.cat((token_ids[:max_len-1], torch.tensor([EOS_IDX])))
    else:
        return token_ids

# function to collate data samples into batch tensors
def collate_fn(batch, text_transform):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch
