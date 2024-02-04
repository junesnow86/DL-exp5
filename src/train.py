import torch
from tqdm import tqdm

from data_utils import PAD_IDX
from evaluate import evaluate


def train(model, 
          optimizer, 
          train_dataloader, 
          val_dataloader,
          loss_fn, 
          device, 
          num_epochs, 
          scheduler=None,
          plot=True,
          figure_file='loss.png',
          model_file='model.pth'):
    model.to(device)
    training_losses = []
    val_losses = []
    min_val_loss = 1e4
    for epoch in range(1, num_epochs+1):
        # Training
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader), desc='training')
        epoch_loss = 0.0
        total_samples = 0
        for src, tgt in progress_bar:
            src = src.long().to(device)
            tgt = tgt.long().to(device)

            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]

            model.train()
            logits = model(src, tgt_input)

            tgt_label_mask = tgt_label != PAD_IDX
            preds = torch.argmax(logits, -1)
            correct = preds == tgt_label
            acc = torch.sum(correct * tgt_label_mask) / torch.sum(tgt_label_mask)

            n, seq_len = tgt_label.shape
            logits = torch.reshape(logits, (n * seq_len, -1))
            tgt_label = torch.reshape(tgt_label, (n * seq_len, ))
            loss = loss_fn(logits, tgt_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Update tqdm progress bar with the latest loss value
            epoch_loss += loss.item() * len(src)
            total_samples += len(src)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

            # Calculate training BLEU score
            # bleu_score_train = evaluate_batch(model, src, tgt, tgt_vocab_transform)
            # progress_bar.set_postfix({'bleu': f'{bleu_score_train:.4f}'})
        training_losses.append(epoch_loss / total_samples)

        # Calculate validation loss
        val_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, total=len(val_dataloader), desc='validation')
            for src, tgt in progress_bar:
                src = src.long().to(device)
                tgt = tgt.long().to(device)

                tgt_input = tgt[:, :-1]
                tgt_label = tgt[:, 1:]

                model.eval()
                logits = model(src, tgt_input)

                tgt_label_mask = tgt_label != PAD_IDX
                preds = torch.argmax(logits, -1)
                correct = preds == tgt_label
                acc = torch.sum(correct * tgt_label_mask) / torch.sum(tgt_label_mask)

                n, seq_len = tgt_label.shape
                logits = torch.reshape(logits, (n * seq_len, -1))
                tgt_label = torch.reshape(tgt_label, (n * seq_len, ))
                loss = loss_fn(logits, tgt_label)

                val_loss += loss.item() * len(src)
                total_samples += len(src)
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
        val_loss = val_loss / total_samples
        val_losses.append(val_loss)
        print(f'Epoch {epoch}/{num_epochs}, Train loss: {training_losses[-1]:.4f}, Val loss: {val_losses[-1]:.4f}')

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), model_file)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(training_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(figure_file)

    return model.to('cpu')


if __name__ == '__main__':
    NUM_EPOCH = 20
    BATCH_SIZE = 32
    LR = 0.0001
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

    # Load vocab
    import os

    from torchtext.data.utils import get_tokenizer

    from data_utils import (
        SRC_LANGUAGE,
        TGT_LANGUAGE,
        collate_fn,
        create_data_iter,
        sequential_transforms,
        tensor_transform,
        truncate_transform,
    )

    vocab_save_dir = '/home/ljt/DL-exp5/vocab/news-commentary-v15'
    # vocab_save_dir = '/home/ljt/DL-exp5/vocab/back-translation'
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
    from torch.utils.data import DataLoader
    train_iter = create_data_iter('/home/ljt/DL-exp5/data/news-commentary-v15/train.tsv')
    # train_iter = create_data_iter('/home/ljt/DL-exp5/data/back-translation/train.tsv')
    train_dataloader = DataLoader(list(train_iter), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  collate_fn=lambda data: collate_fn(data, text_transform))

    val_iter = create_data_iter('/home/ljt/DL-exp5/data/news-commentary-v15/val.tsv')
    # val_iter = create_data_iter('/home/ljt/DL-exp5/data/back-translation/val.tsv')
    val_dataloader = DataLoader(list(val_iter), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  collate_fn=lambda data: collate_fn(data, text_transform))

    test_iter = create_data_iter('/home/ljt/DL-exp5/data/news-commentary-v15/test.tsv')
    # test_iter = create_data_iter('/home/ljt/DL-exp5/data/back-translation/test.tsv')
    test_dataloader = DataLoader(list(test_iter), 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  collate_fn=lambda data: collate_fn(data, text_transform))

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from seq2seq_network import MyTransformer
    model = MyTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, 0, n_layer, n_head, d_model, d_ff, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5 * LR, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCH)

    # print('>>> Start training...')
    # train(
    #     model=model, 
    #     optimizer=optimizer, 
    #     train_dataloader=train_dataloader, 
    #     val_dataloader=val_dataloader,
    #     loss_fn=loss_fn, 
    #     device=DEVICE, 
    #     num_epochs=NUM_EPOCH,
    #     scheduler=scheduler,
    #     wait=3,
    #     plot=True,
    #     figure_file='../figures/news-commentary_loss_02041310.png',
    #     # figure_file='../figures/back-translation_loss_02032200.png',
    #     model_file='../checkpoints/news-commentary_02041310.pth'
    #     # model_file='../checkpoints/back-translation_02032200.pth',
    #     )
    # print(f'>>> Training finished.')

    model.load_state_dict(torch.load('../checkpoints/news-commentary_02041310.pth'))

    # Test
    bleu_score = evaluate(model, test_dataloader, vocab_transform[TGT_LANGUAGE], DEVICE)
    print(f'Validation BLEU score: {bleu_score:.4f}')

    # # Translate
    # from evaluate import translate
    # num_translated = 0
    # results = []
    # test_iter = create_data_iter('/home/ljt/DL-exp5/data/news-commentary-v15/test.tsv')
    # # test_iter = create_data_iter('/home/ljt/DL-exp5/data/back-translation/test.tsv')
    # for src, tgt in tqdm(test_iter, total=50, desc='translating'):
    #     src = [src]
    #     tgt = [tgt]
    #     translated = translate(model, src, text_transform, vocab_transform[TGT_LANGUAGE], DEVICE)
    #     results.append({'src': src[0], 'tgt': tgt[0], 'translated': translated[0]})
    #     num_translated += 1

    #     if num_translated == 50:
    #         break

    # with open('../results/news-commentary_02041310.json', 'w', encoding='utf-8') as file:
    # # with open('../results/back-translation_02032200.json', 'w', encoding='utf-8') as file:
    #     import json
    #     json.dump(results, file, ensure_ascii=False, indent=4)
