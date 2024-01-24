import os
import random
from timeit import default_timer as timer

import numpy as np
import torch
from torchtext.data.utils import get_tokenizer

from build_vocab import (
    SRC_LANGUAGE,
    TGT_LANGUAGE,
    UNK_IDX,
    build_vocab_from_iterator,
    create_data_iter,
    special_symbols,
    yield_tokens,
)
from collate_data import collate_fn, sequential_transforms, tensor_transform
from evaluate import evaluate
from inference import translate
from seq2seq_network import Seq2SeqTransformer
from train import train_epoch

if __name__ == '__main__':
    token_transform = {}
    vocab_transform = {}

    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='zh_core_web_sm')

    # build vocab
    vocab_save_dir = '/storage/1008ljt/DL-exp5/vocab/news-commentary-v15'
    print('building vocab...')
    start_time = timer()
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_file = os.path.join(vocab_save_dir, f'vocab_{ln}.pt')
        if os.path.exists(vocab_file):
            print(f'vocab file {vocab_file} exists, skip building vocab')
            vocab_transform[ln] = torch.load(vocab_file)
            continue

        data_iter = create_data_iter('/storage/1008ljt/DL-exp5/data/news-commentary-v15.en-zh.tsv')
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

    # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                vocab_transform[ln], #Numericalization
                                                tensor_transform) # Add BOS/EOS and create tensor

    # set seed
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # create model
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    # initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=UNK_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Training
    print('>>> Start training...')
    NUM_EPOCHS = 20
    for epoch in range(1, NUM_EPOCHS+1):
        train_iter = create_data_iter('/storage/1008ljt/DL-exp5/data/news-commentary-v15/train.tsv')
        train_loss = train_epoch(transformer, optimizer, train_iter, lambda data: collate_fn(data, text_transform), loss_fn, BATCH_SIZE, DEVICE)

        val_iter = create_data_iter('/storage/1008ljt/DL-exp5/data/news-commentary-v15/val.tsv')
        val_loss = evaluate(transformer, val_iter, lambda data: collate_fn(data, text_transform), loss_fn, BATCH_SIZE, DEVICE)

        print(f'Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}')
    print('>>> Training finished!')

    # Inference
    print(translate(transformer, 'A black dog eats food.', text_transform, vocab_transform))
