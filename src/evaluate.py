import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from data_utils import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    SRC_LANGUAGE,
    TGT_LANGUAGE,
    MAX_LEN,
)

@torch.no_grad()
def ids2sentence(token_ids: torch.Tensor, vocab_transform):
    sentence = " ".join(vocab_transform.lookup_tokens(list(token_ids.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
    return sentence

@torch.no_grad()
def get_batch_references(batch_token_ids, vocab_transform):
    references = []
    for token_ids in batch_token_ids:
        reference = ids2sentence(token_ids, vocab_transform)
        references.append(reference)
    return references

@torch.no_grad()
def translate_helper(model, src_tensor_batch, tgt_tensor_batch):
    '''
    translate a batch of src tensor to tgt tensor
    '''
    model.eval()
    for i in range(MAX_LEN - 1):
        tgt_hat = model(src_tensor_batch, tgt_tensor_batch)
        for j in range(len(src_tensor_batch)):
            tgt_tensor_batch[j, i] = torch.argmax(tgt_hat[j, i-1])
            if torch.equal(tgt_tensor_batch[j, i], torch.tensor(EOS_IDX).to(tgt_tensor_batch.device)):
                break
    return tgt_tensor_batch

@torch.no_grad()
def translate(model, src_sentence_batch, text_transform, tgt_vocab_transform):
    '''
    translate a batch of src sentences to tgt sentences
    '''
    src_tensor_batch = []
    for src_sentence in src_sentence_batch:
        src_tensor = text_transform[SRC_LANGUAGE](src_sentence)
        src_tensor_batch.append(src_tensor)
    src_tensor_batch = torch.stack(src_tensor_batch)
    tgt_tensor_batch = torch.ones(len(src_sentence_batch), MAX_LEN).fill_(BOS_IDX).type(torch.long)

    tgt_tensor_batch = translate_helper(model, src_tensor_batch, tgt_tensor_batch)

    output_sentences = get_batch_references(tgt_tensor_batch, tgt_vocab_transform)
    return output_sentences

@torch.no_grad()
def evaluate_batch(model, src_tensor_batch, tgt_tensor_batch, tgt_vocab_transform):
    '''
    translate a batch of src tensor to tgt tensor
    convert tgt tensor to tgt sentence as reference
    use model to translate logits tensor to hypothesis
    '''
    model.eval()

    references = get_batch_references(tgt_tensor_batch, tgt_vocab_transform)

    tgt_for_hypo = torch.ones(len(tgt_tensor_batch), MAX_LEN).fill_(BOS_IDX).type(torch.long).to(src_tensor_batch.device)
    hypotheses = translate_helper(model, src_tensor_batch, tgt_for_hypo)
    hypotheses = get_batch_references(hypotheses, tgt_vocab_transform)

    # Calculate BLEU score
    assert len(references) == len(hypotheses), f'len(references)={len(references)}, len(hypotheses)={len(hypotheses)}'
    total_score = 0.0
    for reference, hypothesis in zip(references, hypotheses):
        reference = reference.split()
        hypothesis = hypothesis.split()
        score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method4)
        total_score += score

    return total_score / len(references)

@torch.no_grad()
def evaluate(model, val_dataloader, tgt_vocab_transform, device='cuda'):
    '''
    val_dataloader: iter batches of (src tensor, tgt tensor)
    convert tgt tensor to tgt sentence as reference
    use model to translate logits tensor to hypothesis
    '''
    model.to(device)
    model.eval()
    total_score = 0.0

    progress_bar = tqdm(val_dataloader, total=len(val_dataloader), desc='evaluation')
    for src, tgt in progress_bar:
        src = src.long().to(device)
        tgt = tgt.long().to(device)

        references = get_batch_references(tgt, tgt_vocab_transform)

        tgt_for_hypo = torch.ones(len(tgt), MAX_LEN).fill_(BOS_IDX).type(torch.long).to(src.device)
        hypotheses = translate_helper(model, src, tgt_for_hypo)
        hypotheses = get_batch_references(hypotheses, tgt_vocab_transform)

        # Calculate BLEU score
        assert len(references) == len(hypotheses), f'len(references)={len(references)}, len(hypotheses)={len(hypotheses)}'
        for reference, hypothesis in zip(references, hypotheses):
            reference = reference.split()
            hypothesis = hypothesis.split()
            score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method1)
            total_score += score

    return total_score / len(val_dataloader)
