import torch.nn.functional as F
import esm.inverse_folding
from esm.inverse_folding.util import CoordBatchConverter
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import os
from Bio import SeqIO
import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def score_sequence(model, converter, alphabet, coords, seq):
    batch = [(coords, None, seq)]
    coords, confidence, strs, tokens, padding_mask = converter(batch)
    prev_output_tokens = tokens[:, :-1]
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx).to(device)

    logits, _ = model.forward(
        coords.to(device),
        padding_mask.to(device),
        confidence.to(device),
        prev_output_tokens.to(device)
    )

    loss = F.cross_entropy(logits, target.to(device), reduction='none')
    avgloss = torch.sum(loss * ~target_padding_mask, dim=-1) / \
        torch.sum(~target_padding_mask, dim=-1)
    ll_fullseq = -avgloss.detach().cpu().numpy().item()
    coord_mask = torch.all(
        torch.all(torch.isfinite(coords.to(device)), dim=-1), dim=-1)
    coord_mask = coord_mask[:, 1:-1]
    avgloss = torch.sum(loss * coord_mask, dim=-1) / \
        torch.sum(coord_mask, dim=-1)
    ll_withcoord = -avgloss.detach().cpu().numpy().item()
    return ll_fullseq, ll_withcoord


def read_seq_from_fasta(fasta_file):
    for record in SeqIO.parse(fasta_file, "fasta"):
        return str(record.seq)


def full_sequence(seq, mutants):
    for mutant in mutants.split(':'):
        wt, idx, mt = mutant[0], int(mutant[1:-1]), mutant[-1]
        if wt != seq[idx]:
            raise ValueError("WT does not match")
        seq = seq[:idx] + mt + seq[idx+1:]
    return seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--mutant", type=str, required=True)
    parser.add_argument("--pdb", type=str, required=True)
    parser.add_argument("--chain", type=str, default="A")
    args = parser.parse_args()
    pdb_file = args.pdb
    chain = args.chain

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device)
    model = model.eval()
    coords, seq = esm.inverse_folding.util.load_coords(pdb_file, chain)
    assert read_seq_from_fasta(args.fasta) == seq

    batch_converter = CoordBatchConverter(alphabet)
    df = pd.read_csv(args.mutant)

    if_score = []
    for m in tqdm(df['mutant']):
        try:
            m_seq = full_sequence(seq, m)
        except ValueError:
            m_seq = None
        ll, _ = score_sequence(model, batch_converter, alphabet, coords, m_seq)
        if_score.append(ll)
    cols = list(df.columns)
    df['esm_if1_gvp4_t16_142M_UR50'] = if_score
    # keep the columns in the same order, and put the new column at the end
    df = df[cols + ['esm_if1_gvp4_t16_142M_UR50']]
    df.to_csv(args.mutant, index=False)


if __name__ == '__main__':
    main()
