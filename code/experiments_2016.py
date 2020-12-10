"""
This code runs a number of experiments on TREC data in order to determine the
performance of several different information retrieval methods using root- and
rule-based terms.
"""

import csv
import os
import re
import spacy

from bs4 import BeautifulSoup
from gensim import corpora, models, similarities
from gensim.test.utils import get_tmpfile
from parmenides.conf import settings
from parmenides.document import Document, Section
from parmenides.utils import cleanup, get_documents, import_class, init

NUM_TOPICS = 60

nlp = spacy.load('en')

class TrecReader:

    def __init__(self, filename='data/pmc_terms.txt'):

        self.index = []
        self.filename = filename

    def __iter__(self):

        self.index.clear()
        with open(self.filename, 'r') as infile:
            for line in infile:
                words = line.split(' ')
                templates = []
                for word in words[1:]:
                    tree = TermTree.from_term(word)
                    for template in tree.get_templates():
                        templates.append(template)
                words += templates
                self.index.append(words[0])
                yield (words[0], words[1:])

class TrecCorpus:

    def __init__(self, dictionary, filename='data/pmc_terms.txt'):

        self.index = []
        self.dictionary = dictionary
        self.filename = filename

    def __iter__(self):

        reader = TrecReader(self.filename)
        self.index = reader.index

        for doc in reader:
            yield self.dictionary.doc2bow(doc[1])

def main():

    init()
    settings.DOCUMENT_SOURCE = 'trecparse.nxml.NXMLSource'

    #preprocess_samples()

    print("PARMENIDES MODELS.")

    print("Loading dictionary.")
    dictionary = corpora.Dictionary(doc[1] for doc in TrecReader())
    print("Generating LSI model.")
    corpus = TrecCorpus(dictionary)
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
    print("Building index.")
    index_tempfile = get_tmpfile("index")
    index = similarities.Similarity(index_tempfile, lsi[corpus],
            num_features=lsi.num_topics)

    topics = list(get_topics('data/topics2016.xml'))
    processor = import_class(settings.PROCESSOR)()

    print("Evaluating LSI model.")
    evaluate(corpus, topics, processor, dictionary, index, lsi, 'data/PARMLSI16.txt',
        'PARMLSI')

    print("Generating LDA model.")
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
    print("Building index.")
    index_tempfile = get_tmpfile("index")
    index = similarities.Similarity(index_tempfile, lda[corpus],
            num_features=lda.num_topics)

    print("Evaluating LDA model.")
    evaluate(corpus, topics, processor, dictionary, index, lda, 'data/PARMLDA16.txt',
        'PARMLDA')

    print("BASELINE MODELS")

    print("Loading dictionary.")
    dictionary = corpora.Dictionary(doc[1] for doc in\
        TrecReader('data/pmc_words.txt'))
    print("Generating LSI model.")
    corpus = TrecCorpus(dictionary, 'data/pmc_words.txt')
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
    print("Building index.")
    index_tempfile = get_tmpfile("index")
    index = similarities.Similarity(index_tempfile, lsi[corpus],
            num_features=lsi.num_topics)
    print("Evaluating LSI model.")
    baseline(corpus, topics, dictionary, index, lsi, 'data/WORDLSI16.txt', 'WORDLSI')
    print("Generating LDA model.")
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
    print("Building index.")
    index_tempfile = get_tmpfile("index")
    index = similarities.Similarity(index_tempfile, lda[corpus],
            num_features=lda.num_topics)

    print("Evaluating LDA model.")
    baseline(corpus, topics, dictionary, index, lda, 'data/WORDLDA16.txt', 'WORDLDA')

    cleanup()

def evaluate(corpus, topics, processor, dictionary, index, model, filename, run_name):

    with open(filename, 'w') as outfile:
        for topic in topics:
            topic_terms = [str(term) for tree in processor.process(topic) \
                    for term in tree.terms]
            topic_doc = dictionary.doc2bow(topic_terms)
            sims = index[model[topic_doc]]
            sorted_sims = sorted(enumerate(sims), key=lambda item: -item[1])
            rank = 1
            for doc_position, doc_score in sorted_sims[:1000]:
                outfile.write("%s\tQ0\t%s\t%d\t%f\t%s\n" % (
                    topic.identifier,
                    corpus.index[doc_position],
                    rank,
                    doc_score,
                    run_name,
                ))
                rank += 1 

def baseline(corpus, topics, dictionary, index, model, filename, run_name):

    with open(filename, 'w') as outfile:
        for topic in topics:
            topic_words = get_doc_words(topic)
            topic_doc = dictionary.doc2bow(topic_words)
            sims = index[model[topic_doc]]
            sorted_sims = sorted(enumerate(sims), key=lambda item: -item[1])
            rank = 1
            for doc_position, doc_score in sorted_sims[:1000]:
                outfile.write("%s\tQ0\t%s\t%d\t%f\t%s\n" % (
                    topic.identifier,
                    corpus.index[doc_position],
                    rank,
                    doc_score,
                    run_name,
                ))
                rank += 1

def preprocess_samples():

    samples = get_samples(['data/qrels-sampleval-2016.txt',
        'data/qrels-treceval-2016.txt'])
    sample_files = (os.path.join('data', 'pmc2016', str(sample) + '.nxml') \
            for sample in samples)

    processor = import_class(settings.PROCESSOR)()

    with open('data/pmc_terms.txt', 'w') as outfile:
        for document in get_documents(sample_files):
            doc_id = document.identifier
            print("Processing document: %s" % doc_id)
            terms = [str(term) for tree in processor.process(document) \
                    for term in tree.terms]

            outfile.write("%s %s\n" % (doc_id, ' '.join(terms)))

    samples = get_samples(['data/qrels-sampleval-2016.txt',
        'data/qrels-treceval-2016.txt'])
    sample_files = (os.path.join('data', 'pmc2016', str(sample) + '.nxml') \
            for sample in samples)

    with open('data/pmc_words.txt', 'w') as outfile:
        for document in get_documents(sample_files):
            doc_id = document.identifier
            print("Processing document: %s" % doc_id)

            words = get_doc_words(document)

            outfile.write("%s %s\n" % (doc_id, ' '.join(words)))

def get_doc_words(document):

    docstring = document.title
    docs = []
    for section in document.sections:
        if len(docstring) + len(section.content) > 900000:
            docs.append(nlp(docstring))
            if len(section.content) > 900000:
                docstring = "%s %s" % (section.name,
                        section.content[:900000])
            else:
                docstring = "%s %s" % (section.name, section.content)
        else:
            docstring += " %s %s" % (section.name, section.content)
    docs.append(nlp(docstring))
    words = []
    for doc in docs:
        for word in doc:
            if word.is_stop or word.like_url or not word.text.isalnum() \
                    or word.text.strip() == '' or \
                    word.pos in ['PRON', 'PRP', 'PRP$'] or \
                    word.tag in ['PRON', 'PRP', 'PRP$']:
                continue

            words.append(word.lemma_.strip().lower())

    return words

def get_samples(filenames):
    """
    Gets the set of samples used in any number of TREC qrel files.
    """

    samples = set()

    for filename in filenames:
        with open(filename, newline='') as infile:
            reader = csv.reader(infile, delimiter=' ')

            for row in reader:
                samples.add(row[2])

    return samples

def get_topics(topfile, section='summary'):
    """
    Gets the list of topics from an XML file.
    """

    with open(topfile, 'r') as infile:
        soup = BeautifulSoup(infile.read(), 'xml')

        for topic in soup.find_all('topic'):
            number = topic['number']
            topic_type = topic['type']

            note = topic.note.string
            description = topic.description.string
            summary = topic.summary.string

            if section == 'note':
                section_content = note
            elif section == 'description':
                section_content = description
            else:
                section_content = summary

            yield Document(
                identifier=number,
                title=topic_type,
                sections=[Section(name=section, content=section_content)],
                collection='topics',
            )

class TermTree:

    def __init__(self, node, left=None, right=None):

        self.node = node
        self.left = left
        self.right = right
        self.next_node = None

    def get_height(self):

        return self.node if isinstance(self.node, int) else -1

    def to_term(self):

        if isinstance(self.node, int):
            return "%s:%d:%s" % (self.left, self.node, self.right)
        else:
            return self.node

    @classmethod
    def from_term(cls, term):

        regex = re.compile(r':(\d+):')

        queue = regex.split(term)
        for i in range(len(queue)):
            if i % 2 == 0:
                queue[i] = TermTree(queue[i])
            else:
                queue[i] = int(queue[i])
        stack = []

        while len(queue) > 0:
            next_item = queue.pop(0)

            if len(stack) == 0:
                stack.append(next_item)
            elif isinstance(next_item, int):
                stack[-1].next_node = next_item
            else:
                current_height = next_item.get_height()
                next_height = stack[-1].next_node
                if next_height == current_height + 1:
                    new_tree = TermTree(next_height, stack.pop(), next_item)
                    queue.insert(0, new_tree)
                else:
                    stack.append(next_item)

        return stack[0]

    def get_templates(self):

        if isinstance(self.node, int):
            yield "_:%d:%s" % (self.node, self.right)
            yield "%s:%d:_" % (self.left, self.node)
            for template in self.left.get_templates():
                yield template
            for template in self.right.get_templates():
                yield template

if __name__ == "__main__":

    main()
