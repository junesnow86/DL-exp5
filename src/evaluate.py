from torch.utils.data import DataLoader
from tqdm import tqdm

from collate_data import create_mask


def evaluate(model, val_iter, collate_fn, loss_fn, batch_size, device):
    model.eval()
    losses = 0

    # val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(list(val_iter), batch_size=batch_size, collate_fn=collate_fn)

    progress_bar = tqdm(val_dataloader, total=len(val_dataloader))

    for src, tgt in progress_bar:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

        progress_bar.set_postfix({'validation_loss': f'{loss.item()/len(src):.3f}'})

    return losses / len(list(val_dataloader))
