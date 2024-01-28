import os
from timeit import default_timer as timer

import torch
from torchtext.data.utils import get_tokenizer

from data_utils import (
    SRC_LANGUAGE,
    TGT_LANGUAGE,
    UNK_IDX,
    build_vocab_from_iterator,
    create_data_iter,
    special_symbols,
    yield_tokens,
)

if __name__ == '__main__':
    token_transform = {}
    vocab_transform = {}

    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_trf')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='zh_core_web_trf')

    # build vocab
    vocab_save_dir = '/home/ljt/DL-exp5/vocab/news-commentary-v15'
    print('building vocab...')
    start_time = timer()
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_file = os.path.join(vocab_save_dir, f'vocab_{ln}.pt')
        if os.path.exists(vocab_file):
            print(f'vocab file {vocab_file} exists, skip building vocab')
            vocab_transform[ln] = torch.load(vocab_file)
            continue

        data_iter = create_data_iter('/home/ljt/DL-exp5/data/news-commentary-v15/news-commentary-v15.en-zh.tsv')
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(token_transform, data_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
        vocab_transform[ln].set_default_index(UNK_IDX)
        torch.save(vocab_transform[ln], vocab_file)
    end_time = timer()
    print(f'vocab built! time cost: {(end_time - start_time):.3f}s')

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    print(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
