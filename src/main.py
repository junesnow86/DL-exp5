import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from data_utils import PAD_IDX, SRC_LANGUAGE, TGT_LANGUAGE, UNK_IDX, create_data_iter, collate_fn, sequential_transforms, tensor_transform, truncate_transform
from evaluate import evaluate, translate
from train import train
from seq2seq_network import MyTransformer

# Training hyperparameters
NUM_EPOCH = 20
BATCH_SIZE = 8
LR = 0.0001

# Model hyperparameters
MAX_LEN = 150
d_model = 512
d_ff = 2048
n_layer = 6
n_head = 8
dropout = 0.2

# Set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def news_commentary(root_dir='/home/ljt/DL-exp5', load_model_path=None):
    # Load vocab
    vocab_save_dir = os.path.join(root_dir, 'vocab/news-commentary-v15')
    vocab_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_file = os.path.join(vocab_save_dir, f'vocab_{ln}.pt')
        vocab_transform[ln] = torch.load(vocab_file)
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    token_transform = {}
    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_trf')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='zh_core_web_trf')
    # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                vocab_transform[ln], #Numericalization
                                                tensor_transform, # Add BOS/EOS and create tensor
                                                lambda token_ids: truncate_transform(token_ids, MAX_LEN)) # Truncation

    # Load data
    train_iter = create_data_iter(os.path.join(root_dir, 'data/news-commentary-v15/train.tsv'))
    train_dataloader = DataLoader(list(train_iter), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  collate_fn=lambda data: collate_fn(data, text_transform))

    val_iter = create_data_iter(os.path.join(root_dir, 'data/news-commentary-v15/val.tsv'))
    val_dataloader = DataLoader(list(val_iter), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  collate_fn=lambda data: collate_fn(data, text_transform))

    test_iter = create_data_iter(os.path.join(root_dir, 'data/news-commentary-v15/test.tsv'))
    test_dataloader = DataLoader(list(test_iter), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  collate_fn=lambda data: collate_fn(data, text_transform))

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MyTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, 0, n_layer, n_head, d_model, d_ff, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5 * LR, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCH)

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    print('>>> Start training news-commentary-v15...')
    train(
        model=model, 
        optimizer=optimizer, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        loss_fn=loss_fn, 
        device=DEVICE, 
        num_epochs=NUM_EPOCH,
        scheduler=scheduler,
        plot=True,
        figure_file='../figures/news-commentary_loss.png',
        model_file='../checkpoints/news-commentary.pth'
        )
    print(f'>>> Training finished.')

    # Test
    bleu_score = evaluate(model, test_dataloader, vocab_transform[TGT_LANGUAGE], DEVICE)
    print(f'Test BLEU score: {bleu_score:.4f}')

    # Translate
    from evaluate import translate
    num_translated = 0
    results = []
    test_iter = create_data_iter(os.path.join(root_dir, 'data/news-commentary-v15/test.tsv'))
    for src, tgt in tqdm(test_iter, total=50, desc='translating'):
        src = [src]
        tgt = [tgt]
        translated = translate(model, src, text_transform, vocab_transform[TGT_LANGUAGE], DEVICE)
        results.append({'src': src[0], 'tgt': tgt[0], 'translated': translated[0]})
        num_translated += 1

        if num_translated == 50:
            break

    with open('../results/news-commentary.json', 'w', encoding='utf-8') as file:
        import json
        json.dump(results, file, ensure_ascii=False, indent=4)

def back_translation(root_dir='/home/ljt/DL-exp5', load_model_path=None):
    # Load vocab
    vocab_save_dir = os.path.join(root_dir, 'vocab/back-translation')
    vocab_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_file = os.path.join(vocab_save_dir, f'vocab_{ln}.pt')
        vocab_transform[ln] = torch.load(vocab_file)
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    token_transform = {}
    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_trf')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='zh_core_web_trf')
    # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                vocab_transform[ln], #Numericalization
                                                tensor_transform, # Add BOS/EOS and create tensor
                                                lambda token_ids: truncate_transform(token_ids, MAX_LEN)) # Truncation

    # Load data
    train_iter = create_data_iter(os.path.join(root_dir, 'data/back-translation/train.tsv'))
    train_dataloader = DataLoader(list(train_iter), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  collate_fn=lambda data: collate_fn(data, text_transform))

    val_iter = create_data_iter(os.path.join(root_dir, 'data/back-translation/val.tsv'))
    val_dataloader = DataLoader(list(val_iter), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  collate_fn=lambda data: collate_fn(data, text_transform))

    test_iter = create_data_iter(os.path.join(root_dir, 'data/back-translation/test.tsv'))
    test_dataloader = DataLoader(list(test_iter), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  collate_fn=lambda data: collate_fn(data, text_transform))

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MyTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, 0, n_layer, n_head, d_model, d_ff, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5 * LR, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCH)

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    print('>>> Start training back-translation...')
    train(
        model=model, 
        optimizer=optimizer, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        loss_fn=loss_fn, 
        device=DEVICE, 
        num_epochs=NUM_EPOCH,
        scheduler=scheduler,
        plot=True,
        figure_file='../figures/back-translation_loss.png',
        model_file='../checkpoints/back-translation.pth',
        )
    print(f'>>> Training finished.')

    # Test
    bleu_score = evaluate(model, test_dataloader, vocab_transform[TGT_LANGUAGE], DEVICE)
    print(f'Test BLEU score: {bleu_score:.4f}')

    # Translate
    from evaluate import translate
    num_translated = 0
    results = []
    test_iter = create_data_iter(os.path.join(root_dir, 'data/back-translation/test.tsv'))
    for src, tgt in tqdm(test_iter, total=50, desc='translating'):
        src = [src]
        tgt = [tgt]
        translated = translate(model, src, text_transform, vocab_transform[TGT_LANGUAGE], DEVICE)
        results.append({'src': src[0], 'tgt': tgt[0], 'translated': translated[0]})
        num_translated += 1

        if num_translated == 50:
            break

    with open('../results/back-translation.json', 'w', encoding='utf-8') as file:
        import json
        json.dump(results, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    news_commentary()
    back_translation()
