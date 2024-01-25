from torch.utils.data import DataLoader
from tqdm import tqdm

from collate_data import create_mask


def train_epoch(model, optimizer, train_iter, collate_fn, loss_fn, batch_size, device):
    model.train()
    losses = 0
    # train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(list(train_iter), batch_size=batch_size, collate_fn=collate_fn)

    # Wrap the dataloader with tqdm to create a progress bar.
    progress_bar = tqdm(train_dataloader, total=len(train_dataloader))

    for src, tgt in progress_bar:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :].long()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        # Update tqdm progress bar with the latest loss value
        progress_bar.set_postfix({'training_loss': f'{loss.item()/len(src):.3f}'})

    return losses / len(list(train_dataloader))
