import logging
import os
from typing import List, Tuple
import re
import spacy
import torch
from dante_tokenizer import DanteTokenizer
from transformers import AutoModelForTokenClassification, AutoTokenizer


contractions = {
    r"(?<![\w.])no(s)?(?![$\w])": r"em o\g<1>",
    r"(?<![\w.])na(s)?(?![$\w])": r"em a\g<1>",
    r"(?<![\w.])da(s)?(?![$\w])": r"de a\g<1>",
    r"(?<![\w.])do(s)?(?![$\w])": r"de o\g<1>",
    r"(?<![\w.])ao(s)?(?![$\w])": r"a o\g<1>",
    r"(?<![\w.])à(s)?(?![$\w])": r"a a\g<1>",
    r"(?<![\w.])pela(s)?(?![$\w])": r"por a\g<1>",
    r"(?<![\w.])pelo(s)?(?![$\w])": r"por o\g<1>",
    r"(?<![\w.])nesta(s)?(?![$\w])": r"em esta\g<1>",
    r"(?<![\w.])neste(s)?(?![$\w])": r"em este\g<1>",
    r"(?<![\w.])nessa(s)?(?![$\w])": r"em essa\g<1>",
    r"(?<![\w.])nesse(s)?(?![$\w])": r"em esse\g<1>",
    r"(?<![\w.])num(?![$\w])": r"em um",
    r"(?<![\w.])nuns(?![$\w])": r"em uns",
    r"(?<![\w.])numa(s)?(?![$\w])": r"em uma\g<1>",
    r"(?<![\w.])nisso(?![$\w])": r"em isso",
    r"(?<![\w.])naquele(s)?(?![$\w])": r"em aquele\g<1>",
    r"(?<![\w.])naquela(s)?(?![$\w])": r"em aquela\g<1>",
    r"(?<![\w.])naquilo(?![$\w])": r"em aquelo",
    r"(?<![\w.])duma(s)?(?![$\w])": r"de uma\g<1>",
    r"(?<![\w.])daqui(?![$\w])": r"de aqui",
    r"(?<![\w.])dali(?![$\w])": r"de ali",
    r"(?<![\w.])daquele(s)?(?![$\w])": r"de aquele\g<1>",
    r"(?<![\w.])daquela(s)?(?![$\w])": r"de aquela\g<1>",
    r"(?<![\w.])deste(s)?(?![$\w])": r"de este\g<1>",
    r"(?<![\w.])desta(s)?(?![$\w])": r"de esta\g<1>",
    r"(?<![\w.])desse(s)?(?![$\w])": r"de esse\g<1>",
    r"(?<![\w.])dessa(s)?(?![$\w])": r"de essa\g<1>",
    r"(?<![\w.])daí(?![$\w])": r"de aí",
    r"(?<![\w.])dum(?![$\w])": r"de um",
    r"(?<![\w.])donde(?![$\w])": r"de onde",
    r"(?<![\w.])disto(?![$\w])": r"de isto",
    r"(?<![\w.])disso(?![$\w])": r"de isso",
    r"(?<![\w.])daquilo(?![$\w])": r"de aquilo",
    r"(?<![\w.])dela(s)?(?![$\w])": r"de ela\g<1>",
    r"(?<![\w.])dele(s)?(?![$\w])": r"de ele\g<1>",
    r"(?<![\w.])nisto(?![$\w])": r"em isto",
    r"(?<![\w.])nele(s)?(?![$\w])": r"em ele\g<1>",
    r"(?<![\w.])nela(s)?(?![$\w])": r"em ela\g<1>",
    r"(?<![\w.])d'?ele(s)?(?![$\w])": r"de ele\g<1>",
    r"(?<![\w.])d'?ela(s)?(?![$\w])": r"de ela\g<1>",
    r"(?<![\w.])noutro(s)?(?![$\w])": r"em outro\g<1>",
    r"(?<![\w.])aonde(?![$\w])": r"a onde",
    r"(?<![\w.])àquela(s)?(?![$\w])": r"a aquela\g<1>",
    r"(?<![\w.])àquele(s)?(?![$\w])": r"a aquele\g<1>",
    r"(?<![\w.])àquilo(?![$\w])": r"a aquelo",
    r"(?<![\w.])contigo(?![$\w])": r"com ti",
    r"(?<![\w.])né(?![$\w])": r"não é",
    r"(?<![\w.])comigo(?![$\w])": r"com mim",
    r"(?<![\w.])contigo(?![$\w])": r"com ti",
    r"(?<![\w.])conosco(?![$\w])": r"com nós",
    r"(?<![\w.])consigo(?![$\w])": r"com si",
    r"(?<![\w.])pra(?![$\w])": r"para a",
    r"(?<![\w.])pro(?![$\w])": r"para o",
}


def replace_keep_case(word, replacement, text):
    """
    Custom function for replace keeping the original case.
    Parameters
    ----------
    word: str
        Text to be replaced.
    replacement: str
        String to replace word.
    text:
        Text to be processed.
    Returns
    -------
    str:
        Processed string
    """

    def func(match):
        g = match.group()
        repl = match.expand(replacement)
        if g.islower():
            return repl.lower()
        if g.istitle():
            return repl.capitalize()
        if g.isupper():
            return repl.upper()
        return repl

    return re.sub(word, func, text, flags=re.I)


def expand_contractions(text: str) -> str:
    """
    Replace contractions to their based form.
    Parameters
    ----------
    text: str
        Text that may contain contractions.
    Returns
    -------
    str:
        Text with expanded contractions.
    """

    for contraction in contractions.keys():
        replace_str = contractions[contraction]
        text = replace_keep_case(contraction, replace_str, text)

    return text


try:
    nlp = spacy.load("pt_core_news_sm")
except Exception:
    os.system("python -m spacy download pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm")
dt_tokenizer = DanteTokenizer()

default_model = "News"
model_choices = {
    "News": "Emanuel/porttagger-news-base",
}

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Parser:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pre_tokenizer = None
        self.load_model()

    def load_model(self, model_name: str = default_model):
        if model_name not in model_choices.keys():
            logger.error(
                "Selected model is not supported, resetting to the default model.")
            model_name = default_model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_choices[model_name])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_choices[model_name])
        self.pre_tokenizer = nlp


parser = Parser()


def parse_text(text) -> Tuple[List[str], List[str], List[str]]:
    text = expand_contractions(text)
    doc = parser.pre_tokenizer(text)
    tokens = [token.text if not isinstance(
        token, str) else token for token in doc]

    input_tokens = parser.tokenizer(
        tokens,
        return_tensors="pt",
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    output = parser.model(input_tokens["input_ids"])

    i_token = 0
    labels = []
    scores = []
    for off, is_special_token, pred in zip(
        input_tokens["offset_mapping"][0],
        input_tokens["special_tokens_mask"][0],
        output.logits[0],
    ):
        if is_special_token or off[0] > 0:
            continue
        label = parser.model.config.id2label[int(pred.argmax(axis=-1))]
        labels.append(label)
        scores.append(
            "{:.2f}".format(
                100 * float(torch.softmax(pred, dim=-1).detach().max()))
        )
        i_token += 1

    return tokens, labels, scores


# Example usage:
input_text = "A população não poderia ter acesso a relatórios que explicassem, por exemplo, os motivos exatos de atrasos em obras de linhas e estações."
tokens, labels, scores = parse_text(input_text)
for token, label, score in zip(tokens, labels, scores):
    print(f"Token: {token}, Label: {label}, Score: {score}")
