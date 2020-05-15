import nltk
from nltk.corpus import brown

ficbooks = brown.fileids(categories = ['fiction', 'science_fiction'])

nonficbooks = brown.fileids(categories = ['news', 'history', 'government', 'editorial', 'learned'])

for book in nonficbooks:
    outfile = open(book + '.txt', 'w')
    for para in brown.paras(book):
        sents = []
        for sent in para:
            sents.append(' '.join(sent).replace(',',''))
        p = ' '.join(sents)
        outfile.write(p + '\n')
    outfile.close()