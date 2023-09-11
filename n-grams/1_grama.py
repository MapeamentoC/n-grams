import fasttext
import spacy
import re


def load_lexicon(path, tag):
    lexicon = {}
    with open(path, 'r', encoding='utf8') as infile:
        for linha in infile:
            if linha[2] == tag:
                lexicon[linha[0]] = ''

    return lexicon


def load_stop_word(path):
    stop_words = set()
    with open(path, 'r', encoding='utf8') as infile:
        for line in infile:
            stop_words.add(line.strip())

    return stop_words


def verificar_numero(string):
    padrao = r'\d+'  # Expressão regular para encontrar um ou mais dígitos
    correspondencias = re.findall(padrao, string)

    if correspondencias:
        return True  # A string contém um número
    else:
        return False  # A string não contém um número


def validar_pt(sentenca):
    predictions = model.predict(sentenca, k=1)
    if predictions[0][0] == '__label__pt':
        is_pt = True
    else:
        is_pt = False

    return is_pt


model = fasttext.load_model('./utilitarios\\lid.176.ftz')
stop_word = load_stop_word('./utilitarios\\stoplist-portugues.txt')
lexicon = load_lexicon('./portilexicon-ud.tsv', 'ADV')


with open('./1-Gram_results.txt', 'r', encoding='utf8') as infile:
    with open('./utilitarios\\1-Gram_results.txt', 'w', encoding='utf8') as outfile:
        outfile.write('Type\tRank\tFreq\tRange\tNormFreq\tNormRange\n')
        for linha in infile:
            termo = linha.split('\t')[0]
            if validar_pt(termo) == True and verificar_numero(termo) == False:
                if termo not in lexicon:
                    if termo not in stop_word:
                        outfile.write(linha)
