import argparse

from rouge import Rouge
from transformers import AutoTokenizer


class BertRouge:
    EOS_TOKEN = '[unused1]'
    # FIXME: In Japanese, [CLS] token becomes EOS token which has EOS token id in English BERT model.
    EOS_TOKEN_ID = 2

    def __init__(self, model_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._rouge = Rouge()

    def get_scores(self, hypothesis: list, references: list, avg: bool = False) -> (list or dict):
        hyps = [
            ' '.join(self._filter_hypothesis(self._tokenizer.tokenize(h)))
            for h in hypothesis
        ]

        refs = [
            ' '.join(self._tokenizer.tokenize(r))
            for r in references
        ]

        return self._rouge.get_scores(hyps, refs, avg=avg)

    def _filter_hypothesis(self, tokens: list) -> list:
        ids = self._tokenizer.convert_tokens_to_ids(tokens)
        try:
            eos_index = ids.index(self.EOS_TOKEN_ID)
        except Exception:
            eos_index = None
        return tokens[:eos_index]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        default='cl-tohoku/'
                        'bert-base-japanese-whole-word-masking')
    parser.add_argument('-c', '--candidate')
    parser.add_argument('-g', '--gold')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.candidate) as f:
        hypothesis = [line for line in f]

    with open(args.gold) as f:
        references = [line for line in f]

    bert_rouge = BertRouge(args.model)
    print(bert_rouge.get_scores(hypothesis, references, avg=True))


if __name__ == '__main__':
    main()
