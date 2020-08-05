import argparse
from typing import Tuple

import pandas as pd
from rouge import Rouge
import torch
from transformers import AutoTokenizer


class GreedySelector:
    def __init__(self, n: int = 3):
        self._n = n
        self._rouge = Rouge()

    def __call__(self, doc_sent_list: list, abstract_sent_list: list) -> list:
        max_rouge = 0.0
        selected = []
        for _ in range(self._n):
            cur_max_rouge = max_rouge
            cur_id = -1
            for i in range(len(doc_sent_list)):
                if (i in selected):
                    continue
                c = selected + [i]
                hypothesis = '\n'.join([
                    ' '.join(doc_sent_list[idx])
                    for idx in c
                ])
                reference = '\n'.join([
                    ' '.join(tokens)
                    for tokens in abstract_sent_list
                ])
                score = self._rouge.get_scores(hypothesis, reference)
                rouge_1 = score[0]['rouge-1']['f']
                rouge_2 = score[0]['rouge-2']['f']
                rouge_score = rouge_1 + rouge_2
                if rouge_score > cur_max_rouge:
                    cur_max_rouge = rouge_score
                    cur_id = i
            if (cur_id == -1):
                return selected
            selected.append(cur_id)
            max_rouge = cur_max_rouge

        return sorted(selected)


class BertProcessor:
    CLS_TOKEN = '[CLS]'
    SEP_TOKEN = '[SEP]'

    def __init__(self, tokenizer: AutoTokenizer):
        self._tokenizer = tokenizer

    @staticmethod
    def subtokens2tokens(subtokens: list) -> list:
        tokens = []
        temp = []
        for subtoken in subtokens:
            if not temp:
                temp.append(subtoken)
                continue

            if subtoken.startswith('##'):
                temp.append(subtoken[2:])
                continue

            tokens.append(''.join(temp))
            temp = [subtoken]

        if temp:
            tokens.append(''.join(temp))

        return tokens

    def tokenize(self, text: str, enable_subtoken: bool = True) -> Tuple[list, list]:
        tokens = self._tokenizer.tokenize(text)
        if not enable_subtoken:
            tokens = BertProcessor.subtokens2tokens(tokens)

        token_idxs = self._tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_idxs


class JapaneseBertProcessor(BertProcessor):
    @staticmethod
    def text2sents(text: str) -> list:
        sents = []
        for sent in text.split('\n'):
            sub_sents = sent.split('。')
            for i, sub_sent in enumerate(sub_sents):
                if not sub_sent:
                    continue

                s = sub_sent
                if i != len(sub_sents) - 1 or sent.endswith('。'):
                    s += '。'
                sents.append(s)

        return sents


class PreSummDataConverter:
    SRC_META_TOKENS = ('[CLS]', '[SEP]', '[SEP] [CLS]')
    TGT_META_TOKENS = ('[unused0]', '[unused1]', '[unused2]')

    def __init__(self,
                 model_name: str = 'cl-tohoku/bert-base-japanese-whole-word-masking',
                 n: int = 3):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_tokens(['[unused0]', '[unused1]', '[unused2]'])
        tokenizer.vocab['[unused0]'] = 32000
        tokenizer.vocab['[unused1]'] = 32001
        tokenizer.vocab['[unused2]'] = 32002

        jb_processor = JapaneseBertProcessor(tokenizer)

        greedy_selector = GreedySelector(n)

        self.greedy_selector = greedy_selector
        self.jb_processor = jb_processor
        self.tokenizer = tokenizer
        self.CLS_VID = self.tokenizer.vocab[jb_processor.CLS_TOKEN]
        self.SEP_VID = self.tokenizer.vocab[jb_processor.SEP_TOKEN]

    def __call__(self, raw_src: str, raw_tgt: str):
        data = {}
        data.update(self._convert_src(raw_src))
        data.update(self._convert_tgt(raw_tgt))
        data['src_sent_labels'] = self._get_greedy_sents(raw_src, raw_tgt)
        return data

    def _convert_src(self, text: str) -> dict:
        src_txt = self.jb_processor.text2sents(text)
        berttext = self.convert2berttext(src_txt, self.SRC_META_TOKENS)
        _, src_subtoken_idxs = self.jb_processor.tokenize(berttext)
        return dict(
            src=src_subtoken_idxs,
            segs=self.get_segs(src_subtoken_idxs),
            clss=self.get_clss(src_subtoken_idxs),
            src_txt=src_txt
        )

    def _convert_tgt(self, text: str) -> dict:
        tgt_txt = self.jb_processor.text2sents(text)
        berttext = self.convert2berttext(tgt_txt, self.TGT_META_TOKENS)
        _, tgt_subtoken_idxs = self.jb_processor.tokenize(berttext)
        return dict(
            tgt=tgt_subtoken_idxs,
            tgt_txt='<q>'.join(tgt_txt)
        )

    def _get_greedy_sents(self, raw_src: str, raw_tgt: str) -> list:
        src_sents = [
            self.jb_processor.tokenize(sent, False)[0]
            for sent in self.jb_processor.text2sents(raw_src)
        ]
        tgt_sents = [
            self.jb_processor.tokenize(sent, False)[0]
            for sent in self.jb_processor.text2sents(raw_tgt)
        ]

        sent_labels = self.greedy_selector(src_sents, tgt_sents)
        sent_indicator = [0] * len(src_sents)
        for i in sent_labels:
            sent_indicator[i] = 1
        return sent_indicator

    @staticmethod
    def convert2berttext(sents: list, meta_tokens: tuple) -> str:
        prefix_token, suffix_token, interpolate_token = meta_tokens
        return f'{prefix_token} ' + f' {interpolate_token} '.join(sents) + f' {suffix_token}'

    def get_segs(self, idxs: list) -> list:
        _segs = [-1] + [
            i
            for i, t in enumerate(idxs)
            if t == self.SEP_VID
        ]
        segs = [
            _segs[i] - _segs[i - 1]
            for i in range(1, len(_segs))
        ]

        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        return segments_ids

    def get_clss(self, idxs: list) -> list:
        return [
            i
            for i, t in enumerate(idxs)
            if t == self.CLS_VID
        ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file')
    parser.add_argument('-sc', '--src-col')
    parser.add_argument('-tc', '--tgt-col')
    return parser.parse_args()


def main():
    args = parse_args()
    # Assume ndjson format
    df = pd.read_json(args.input_file, orient='record', lines=True)

    converter = PreSummDataConverter()
    dataset = df.apply(lambda x: converter(getattr(x, args.src_col),
                                           getattr(x, args.tgt_col)),
                       axis=1)
    torch.save(dataset, 'bert_data/jp.all.bert.pt')
