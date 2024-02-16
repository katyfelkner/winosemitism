import os
import csv
import json
import math
import torch
import argparse
import difflib
import logging
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelWithLMHead
from collections import defaultdict
from tqdm import tqdm

# need to have a constant order for ease of printing out
IDENTITY_LIST = ['LGBTQ', 'Queer', 'Transgender', 'Nonbinary', 'Bisexual', 'Pansexual', 'Lesbian', 'Asexual', 'Gay']

def get_log_prob_unigram(masked_token_ids, lm):
    """
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """
    
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    # get model hidden states
    output = model(masked_token_ids)
    hidden_states = output[0]
    return hidden_states


def get_span(seq1, seq2):
    """
    This function extract spans that are shared between two sequences.
    """

    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2


def mask_unigram(data, lm, n=1):
    """
    Score each sentence by masking one word at a time.
    The score for a sentence is the sum of log probability of each word in
    the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    sent1, sent2 = data["stereo"], data["counter"]

    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    # tokenize
    sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
    sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')
    
    # get spans of non-changing tokens
    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])

    assert len(template1) == len(template2)

    N = len(template1)  # num. of tokens that can be masked
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    
    # construct batch
    sent1_batch = torch.empty((N-2, len(sent1_token_ids[0])), dtype=torch.int64)
    sent2_batch = torch.empty((N-2, len(sent2_token_ids[0])), dtype=torch.int64)

    # skipping CLS and SEP tokens, they'll never be masked
    for i in range(1, N-1):
        sent1_masked_token_ids = sent1_token_ids.clone().detach()
        sent2_masked_token_ids = sent2_token_ids.clone().detach()

        sent1_masked_token_ids[0][template1[i]] = mask_id
        sent2_masked_token_ids[0][template2[i]] = mask_id

        sent1_batch[i-1] = sent1_masked_token_ids[0]
        sent2_batch[i-1] = sent2_masked_token_ids[0]

    # send the whole batch to GPU
    hs1 = get_log_prob_unigram(sent1_batch, lm)
    hs2 = get_log_prob_unigram(sent2_batch, lm)

    
    # sum up scores for each masked token
    sent1_sum_log_probs = torch.sum(torch.tensor([log_softmax(hs1)[i-1][template1[i]][sent1_token_ids[0][template1[i]]] for i in range(1, N-1)]))
    sent2_sum_log_probs = torch.sum(torch.tensor([log_softmax(hs2)[i-1][template2[i]][sent2_token_ids[0][template2[i]]] for i in range(1, N-1)]))

    
    score = {}
    # average over iterations
    score["sent1_score"] = sent1_sum_log_probs.item()
    score["sent2_score"] = sent2_sum_log_probs.item()

    return score


def evaluate(args):
    """
    Evaluate a masked language model using CrowS-Pairs dataset.
    """

    print("Evaluating:")
    print("Input:", args.input_file)
    print("Model:", args.lm_model_path)
    print("=" * 100)

    logging.basicConfig(level=logging.INFO)

    # load data into panda DataFrame
    df_data = pd.read_csv(args.input_file)
    # columns: identity, sent_x, sent_y
    # sent_x is "more stereotypical" and sent_y is "less stereotypical"

    # fairly hacky handling of filenames - could fix by reading config file instead of hard coding for my file structure
    # deal with trailing slash
    if args.lm_model_path[-1] == '/': args.lm_model_path = args.lm_model_path[:-1] 
    base_model_path = "../new_finetune/pretrained/" + args.lm_model_path.split('/')[-1].split("-finetuned")[0]
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if hasattr(tokenizer, 'do_lower_case'):
        uncased = tokenizer.do_lower_case
    else:
        uncased = False
    if "opt" in args.lm_model_path:
        model = AutoModelForCausalLM.from_pretrained(args.lm_model_path)
    elif "gpt2" in args.lm_model_path or "bloom" in args.lm_model_path or "bart" in args.lm_model_path:
        model = AutoModelWithLMHead.from_pretrained(args.lm_model_path)
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.lm_model_path)    
        
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    mask_token = tokenizer.mask_token
    log_softmax = torch.nn.LogSoftmax(dim=2)
    vocab = tokenizer.get_vocab()
    with open(args.lm_model_path + ".vocab", "w") as f:
        f.write(json.dumps(vocab))

    lm = {"model": model,
          "tokenizer": tokenizer,
          "mask_token": mask_token,
          "log_softmax": log_softmax,
          "uncased": uncased
    }

    # score each sentence. 
    # each row in the dataframe has the sentid and score for pro and anti stereo.
    df_score = pd.DataFrame(columns=['sent_more', 'sent_less', 
                                     'sent_more_score', 'sent_less_score',
                                     'score', 'bias_target_group'])


    total_pairs = 0
    stereo_score =  0

    # dict for keeping track of scores by category
    category_scores = {group: {'count': 0, 'score': 0, 'metric': None} for group in df_data.target_ID.unique()}

    N = 0
    neutral = 0
    total = len(df_data.index)
    with tqdm(total=total) as pbar:
        for index, data in df_data.iterrows():
            bias = data['target_ID']
            score = mask_unigram(data, lm)

            # round all scores to 3 places
            for stype in score.keys():
                score[stype] = round(score[stype], 3)

            N += 1
            category_scores[bias]['count'] += 1
            pair_score = 0
            if score['sent1_score'] == score['sent2_score']:
                neutral += 1
            else:
                total_pairs += 1
                if score['sent1_score'] > score['sent2_score']:
                        stereo_score += 1
                        category_scores[bias]['score'] += 1
                        pair_score = 1
                        
            sent_more = data['stereo']
            sent_less = data['counter']
            sent_more_score = score['sent1_score']
            sent_less_score = score['sent2_score']

            df_score = pd.concat((df_score, pd.Series({'sent_more': sent_more,
                                        'sent_less': sent_less,
                                        'sent_more_score': sent_more_score,
                                        'sent_less_score': sent_less_score,
                                        'score': pair_score,
                                        'bias_target_group': bias
                                        })), ignore_index=True)

            if index % 10 == 0:
                # write to df every 100 rows
                pbar.update(10)

    df_score.to_csv(args.output_file)
    if args.summary_file:
        summary_path = args.summary_file
    else:
        summary_path = args.output_file + ".summary"

    
    with open(summary_path, 'w') as f:
        f.write('Total examples: ' + str(N) + '\n')
        f.write("Num. neutral:" + str(neutral) + ", % neutral: " + str(round(neutral / N * 100, 2)) + '\n')
        f.write('Winoqueer Overall Score: ' + str(round(stereo_score / N * 100, 2)) + '\n')
        f.write('Score Breakdown by Target of Bias:\n')
        for k, v in category_scores.items():
            f.write("Category: " + k + '\n')
            f.write("    Number of examples: " + str(v['count']) + '\n')
            if v['count'] > 0:
                v['metric'] = round(v['score'] / v['count'] * 100, 2)
                f.write("    Bias score against group " + k + ": " + str(v['metric']) + '\n')

        f.write("For pasting into spreadsheet (Order Overall, " + ", ".join(IDENTITY_LIST) + "):" )
        # use list of keys instead of category_scores.items() to force order to match the spreadsheet
        f.write(str(round(stereo_score / N * 100, 2)) + ", " + ", ".join([str(category_scores[key]['metric']) for key in IDENTITY_LIST]))

    print('=' * 100)
    print("Output written to: " + args.output_file)
    print("summary stats written to: " + summary_path)
    print("For pasting into spreadsheet (Order Overall, " + ", ".join(IDENTITY_LIST) + "):" )
    # use list of keys instead of category_scores.items() to force order to match the spreadsheet
    print(str(round(stereo_score / N * 100, 2)) + ", " + ", ".join([str(category_scores[key]['metric']) for key in IDENTITY_LIST]) + "\n")
    print('=' * 100)


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="path to input file")
parser.add_argument("--lm_model_path", type=str, help="path to pretrained LM model to use")
parser.add_argument("--output_file", type=str, help="path to output file with sentence scores")
parser.add_argument("--summary_file", type=str, help="path to output summary stats", required=False)

args = parser.parse_args()
evaluate(args)
