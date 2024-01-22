import csv
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'zh'

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
    data_iter = create_data_iter('/storage/1008ljt/DL-exp5/data/news-commentary-v15.en-zh.tsv')
    token_transform = {}
    vocab_transform = {}
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(token_transform, data_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
        vocab_transform[ln].set_default_index(0)

    print(vocab_transform[SRC_LANGUAGE].get_itos()[5])
