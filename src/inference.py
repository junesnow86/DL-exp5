import torch

from data_utils import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    SRC_LANGUAGE,
    TGT_LANGUAGE,
    MAX_LEN,
    generate_square_subsequent_mask,
)


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str, text_transform, vocab_transform):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

def translate(model, src_sentence, text_transform, vocab_transform, device):
    src = text_transform[SRC_LANGUAGE](src_sentence).view(1, -1).to(device)
    tgt_input = torch.ones(1, MAX_LEN).fill_(PAD_IDX).type(torch.long).to(device)
    tgt_input[0][0] = BOS_IDX

    model.eval()
    with torch.no_grad():
        for i in range(MAX_LEN - 1):
            tgt_hat = model(src, tgt_input)
            tgt_input[0][i] = torch.argmax(tgt_hat[0][i-1])

    output_sentence = " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_input[0].cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
    print(output_sentence)
