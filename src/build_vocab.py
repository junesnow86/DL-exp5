import csv
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'zh'

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

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

if __name__ == '__main__':
    token_transform = {}
    vocab_transform = {}

    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='zh_core_web_sm')

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        data_iter = create_data_iter('/storage/1008ljt/DL-exp5/data/news-commentary-v15.en-zh.tsv')
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(token_transform, data_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
        vocab_transform[ln].set_default_index(UNK_IDX)

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    print(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
