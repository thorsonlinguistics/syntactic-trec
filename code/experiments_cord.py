"""
This code runs a number of experiments on the TREC-Covid data in order to
determine the performance of several different information retrieval methods
using root- and rule-based terms.
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

NUM_TOPICS = 100

nlp = spacy.load('en')

class FileDescription:

    def __init__(self, cord_uid, title, abstract=None, filename=None):

        self.cord_uid = cord_uid
        self.title = title
        self.abstract = abstract
        self.filename = filename

class TrecReader:

    def __init__(self, filename='data/cord_terms.txt'):

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

    def __init__(self, dictionary, filename='data/cord_terms.txt'):

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
    settings.DOCUMENT_SOURCE = 'trecparse.cord.Cord19Source'

    preprocess_samples()

    print("PARMENIDES MODELS")
    print("Loading dictionary.")
    dictionary = corpora.Dictionary(doc[1] for doc in TrecReader())
    print("Generating LSI model.")
    corpus = TrecCorpus(dictionary)
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
    print("Building index.")
    index_tempfile = get_tmpfile("index")
    index = similarities.Similarity(index_tempfile, lsi[corpus],
            num_features=lsi.num_topics)

    topics = list(get_topics('data/topics-rnd5.xml'))
    processor = import_class(settings.PROCESSOR)()

    print("Evaluating LSI model.")
    evaluate(corpus, topics, processor, dictionary, index, lsi, 'data/PARMLSICORD.txt',
            'PARMLSI')

    print("Generating LDA model.")
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
    print("Building index.")
    index_tempfile = get_tmpfile("index")
    index = similarities.Similarity(index_tempfile, lda[corpus],
            num_features=lda.num_topics)

    print("Evaluating LDA model.")
    evaluate(corpus, topics, processor, dictionary, index, lda, 'data/PARMLDACORD.txt',
            'PARMLDA')

    print("BASELINE MODELS")
    print("Loading dictionary.")
    dictionary = corpora.Dictionary(doc[1] for doc in \
            TrecReader('data/cord_words.txt'))
    print("Generating LSI model.")
    corpus = TrecCorpus(dictionary, 'data/cord_words.txt')
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
    print("Building index.")
    index_tempfile = get_tmpfile("index")
    index = similarities.Similarity(index_tempfile, lsi[corpus],
            num_features=lsi.num_topics)
    print("Evaluating LSI model.")
    baseline(corpus, topics, dictionary, index, lsi, 'data/WORDLSICORD.txt',
            'WORDLSI')
    print("Generating LDA model.")
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
    print("Building index.")
    index_tempfile = get_tmpfile("index")
    index = similarities.Similarity(index_tempfile, lda[corpus],
            num_features=lda.num_topics)
    print("Evaluating LDA model.")
    baseline(corpus, topics, dictionary, index, lda, 'data/WORDLDACORD.txt', 'WORDLDA')

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

    metadata = get_metadata('data/2020-07-16/metadata.csv')

    docids = get_cord_samples(['data/docids-rnd5.txt'])
    sample_files = get_sample_files(metadata, docids)

    processor = import_class(settings.PROCESSOR)()

    with open('data/cord_terms.txt', 'w') as outfile:
        for document in get_documents(sample_files):
            doc_id = document.identifier
            print("Processing document: %s" % doc_id)
            terms = [str(term) for tree in processor.process(document) \
                    for term in tree.terms]

            outfile.write("%s %s\n" % (doc_id, ' '.join(terms)))

    docids = get_cord_samples(['data/docids-rnd5.txt'])
    sample_files = get_sample_files(metadata, docids)

    with open('data/cord_words.txt', 'w') as outfile:
        for document in get_documents(sample_files):
            doc_id = document.identifier
            print("Processing document: %s" % doc_id)

            words = get_doc_words(document)

            outfile.write("%s %s\n" % (doc_id, ' '.join(words)))

def get_doc_words(document):

    docstring = document.title
    for section in document.sections:
        docstring += " %s %s" % (section.name, section.content)
    doc = nlp(docstring[:900000])
    words = []
    for word in doc:
        if word.is_stop or word.like_url or not word.text.isalnum() \
                or word.text.strip() == '' or \
                word.pos in ['PRON', 'PRP', 'PRP$'] or \
                word.tag in ['PRON', 'PRP', 'PRP$']:
            continue

        words.append(word.lemma_.strip().lower())

    return words

def get_cord_samples(filenames):

    samples = set()

    for filename in filenames:
        with open(filename, 'r') as infile:
            for line in infile:
                samples.add(line.strip())

    return samples

def get_topics(topfile, section='narrative'):

    with open(topfile, 'r') as infile:
        soup = BeautifulSoup(infile.read(), 'xml')

        for topic in soup.find_all('topic'):
            number = topic['number']

            query = topic.query.string
            question = topic.question.string
            narrative = topic.narrative.string

            if section == 'query':
                section_content = query
            elif section == 'question':
                section_content = question
            else:
                section_content = narrative

            yield Document(
                identifier=number,
                title=query,
                sections=[Section(name=section, content=section_content)],
                collection='topics',
            )

def get_metadata(metafile):

    docs = {}

    with open(metafile, newline='') as infile:
        reader = csv.DictReader(infile)

        for line in reader:
            docs[line['cord_uid']] = line

    return docs

def get_sample_files(metadata, docids):

    for docid in docids:
        meta = metadata[docid]

        pdf_json_files = [doc.strip() for doc in \
            meta['pdf_json_files'].split(';') if doc.strip()]
        pmc_json_files = [doc.strip() for doc in \
            meta['pmc_json_files'].split(';') if doc.strip()]

        if len(pdf_json_files) >= 1:
            yield FileDescription(
                docid,
                meta['title'],
                meta['abstract'],
                filename=os.path.join('data/2020-07-16', pdf_json_files[0]),
            )
        elif len(pmc_json_files) >= 1:
            yield FileDescription(
                docid,
                meta['title'],
                meta['abstract'],
                filename=os.path.join('data/2020-07-16', pmc_json_files[0]),
            )
        else:
            yield FileDescription(
                docid,
                meta['title'],
                meta['abstract'],
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

    def get_templates(self, depth=0):

        if depth >= 3:
            pass
        elif isinstance(self.node, int):
            yield "_:%d:%s" % (self.node, self.right)
            yield "%s:%d:_" % (self.left, self.node)
            for template in self.left.get_templates(depth+1):
                yield template
            for template in self.right.get_templates(depth+1):
                yield template

if __name__ == "__main__":

    main()
