import torch
from tqdm import tqdm

from data_utils import PAD_IDX


def train(model, optimizer, train_dataloader, loss_fn, device, num_epochs, scheduler=None):
    for epoch in range(1, num_epochs+1):
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader), desc='training')
        for src, tgt in progress_bar:
            src = src.long().to(device)
            tgt = tgt.long().to(device)

            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]

            model.train()
            logits = model(src, tgt_input)

            tgt_label_mask = tgt_label != PAD_IDX
            preds = torch.argmax(logits, -1)
            correct = (preds == tgt_label).float()
            acc = torch.sum(correct * tgt_label_mask) / torch.sum(tgt_label_mask)

            n, seq_len = tgt_label.shape
            logits = logits.reshape(n * seq_len, -1)
            tgt_label = tgt_label.reshape(n * seq_len)
            loss = loss_fn(logits, tgt_label)

            if scheduler is not None:
                scheduler.step(loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update tqdm progress bar with the latest loss value
            progress_bar.set_postfix({'loss': f'{loss.item()/len(src):.4f}'})


if __name__ == '__main__':
    NUM_EPOCH = 1
    BATCH_SIZE = 16
    LR = 0.01
    MAX_LEN = 120
    d_model = 512
    d_ff = 2048
    n_layer = 6
    n_head = 8
    dropout = 0.1

    # Set seed
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Tokenizers
    from torchtext.data.utils import get_tokenizer

    from data_utils import (
        SRC_LANGUAGE,
        TGT_LANGUAGE,
        collate_fn,
        create_data_iter,
        sequential_transforms,
        tensor_transform,
        truncate_transform
    )

    token_transform = {}
    token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_trf')
    token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='zh_core_web_trf')

    # Load vocab
    import os
    vocab_save_dir = '/home/ljt/DL-exp5/vocab/news-commentary-v15'
    vocab_transform = {}
    for ln in ['en', 'zh']:
        vocab_file = os.path.join(vocab_save_dir, f'vocab_{ln}.pt')
        vocab_transform[ln] = torch.load(vocab_file)
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                vocab_transform[ln], #Numericalization
                                                lambda token_ids: truncate_transform(token_ids, MAX_LEN-1), # Truncation
                                                tensor_transform) # Add BOS/EOS and create tensor

    # Load data
    from torch.utils.data import DataLoader
    train_iter = create_data_iter('/home/ljt/DL-exp5/data/news-commentary-v15/train.tsv')
    train_dataloader = DataLoader(list(train_iter), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  collate_fn=lambda data: collate_fn(data, text_transform))

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from seq2seq_network import MyTransformer
    model = MyTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, PAD_IDX, n_layer, n_head, d_model, d_ff, dropout).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95, verbose=True)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # train(model, optimizer, train_dataloader, loss_fn, DEVICE, NUM_EPOCH, scheduler=scheduler)

    from inference import translate
    src_sentence = 'The school lunch program is the largest discrete market for low-cost, healthy food.'
    translate(model, src_sentence, text_transform, vocab_transform, DEVICE)
