"""
This script runs a text retrieval experiment on TREC data. The experiment loads
data from a data source, analyzes it, and trains a topic model. The topic model
is then evaluated against a set of queries, which are also analyzed and
converted into the topic model's vector space. 
"""

import argparse

from gensim import corpora, models, similarities
from gensim.test.utils import get_tmpfile
from parmenides.conf import settings
from parmenides.document import Document, Section
from parmenides.utils import cleanup, import_class, init

parser = argparse.ArgumentParser(description='Run a TREC experiment.')
parser.add_argument('filenames', metavar='FILE', nargs='+', 
        help="a file containing preprocessed data to parse")
parser.add_argument('num_topics', metavar='N', type=int,
        help="the number of topics to generate")
parser.add_argument('tmpfile', metavara='FILE', default='index.tmp',
        help="the name of the temporary index file")
parser.add_argument('topicfile', metavar='FILE',
        help="the path to the file containing TREC topics")
parser.add_argument('topic-type', choices=['cds', 'covid'],
        help="the type of topics under consideration")
parser.add_argument('lsi-file', metavar='FILE', 
        help="the path to the output file for the LSI judgments.")
parser.add_argument('lsi-run', metavar='NAME',
        help="a name for the LSI run.")
parser.add_argument('lda-file', metavar='FILE',
        help="the path to the output file for the LDA judgments.")
parser.add_argument('lsi-run', metavar='NAME',
        help="a name for the LDA run")
parser.add_argument('processor', choices=['parmenides', 'spacy'],
        help="how to process TREC topics into documents")
parser.add_argument('settings-file', metavar='FILE', 
        help="the path to the Parmenides settings file," \
            " for the Parmenides processor.")

nlp = None
parmenides_processor = None

def main():

    print("Building dictionary...")
    dictionary = corpora.Dictionary(doc[1] for doc in
            TrecReader(config.filenames))
    print("Generating LSI model...")
    corpus = TrecCorpus(config.filenames, dictionary)
    lsi = models.LsiModel(corpus, id2word=dictionary,
            num_topics=config.num_topics)
    print("Building index...")
    index_tempfile = get_tmpfile(config.tmpfile)
    index = similarities.Similarity(index_tempfile, lsi[corpus],
            num_features=lsi.num_topics)
    topics = list(get_topics(config.topicfile, config.topic_type))
    print("Evaluating LSI model...")
    evaluate(corpus, topics, dictionary, index, lsi, config.lsi_file,
            config.lsi_run, config.processor, config.settings_file)
    print("Generating LDA model...")
    lda = models.LdaModel(corpus, id2word=dictionary,
            num_topics=config.num_topics)
    print("Building index...")
    index_tempfile = get_tmpfile(config.tmpfile)
    index = similarities.Similarity(index_tempfile, lda[corpus],
            num_features=lda.num_topics)
    print("Evaluating LDA model...")
    evaluate(corpus, topics, dictionary, index, lda, config.lda_file,
            config.lda_run, config.processor, config.settings_file)

    if parmenides_processor:
        cleanup()

class TrecCorpus:
    """
    An iterator over documents in a simple text corpus. The corpus consists of
    space-separated files wherein each line represents a document. The first
    word of each line assigns a unique identifier to a document; the remaining
    words represent tokens in the corpus. Each returned document is a tuple
    containing the document ID and the list of tokens.
    """

    def __init__(self, filenames):

        self.index = []
        self.filenames = filenames

    def __iter__(self):

        self.index.clear()
        for filename in self.filenames:
            with open(filename, 'r') as infile:
                for line in infile:
                    words = line.split(' ')
                    self.index.append(words[0])
                    yield (words[0], words[1:])

class TrecBOW:
    """
    An iterator over bags of words representing documents in a corpus. This
    uses the gensim BOW model and thus requires a gensim dictionary to be
    initialized.
    """

    def __init__(self, dictionary, reader):

        self.dictionary = dictionary
        self.reader = reader

    def __iter__(self):

        for doc in reader:
            yield self.dictionary.doc2bow(doc[1])

def get_topics(topfile, topic_type, topic_selector):
    """
    Gets TREC topics from an XML file. Since the nature of topics varies from
    one TREC corpus to another, the topic type must be provided, naming a
    supported topic format. Topics are returned as Parmenides documents.
    """

    if topic_type == 'cds':
        get_cds_topics(topfile, topic_selector)
    elif topic_type == 'covid':
        get_covid_topics(topfile, topic_selector)
    else:
        raise TypeError("Unsupported topic type: %s" % topic_type)

def get_cds_topics(topfile, topic_selector):
    """
    Gets the topics from the TREC CDS track.
    """

    with open(topfile, 'r') as infile:
        soup = BeautifulSoup(infile.read(), 'xml')

        for topic in soup.find_all('topic'):
            number = topic['number']
            topic_type = topic['type']

            note = topic.note.string
            description = topic.description.string
            summary = topic.summary.string

            if topic_selector == 'note':
                section_content = note
            elif topic_selector == 'description':
                section_content = description
            elif topic_selector == 'summary':
                section_content = summary
            else:
                raise TypeError("Unsupported topic selector: %s" \
                        % topic_selector)

            yield Document(
                identifier=number,
                title=topic_type,
                sections=[Section(name=section, content=section_content)]<
                collection='topics',
            )

def get_covid_topics(topfile, topic_selector):
    """
    Gets the topics from the TREC-COVID dataset.
    """

    with open(topfile, 'r') as infile:
        soup = BeautifulSoup(infile.read(), 'xml')

        for topic in soup.find_all('topic'):
            number = topic['number']

            query = topic.query.string
            question = topic.question.string
            narrative = topic.narrative.string

            if topic_selector == 'query':
                section_content = query
            elif topic_selector == 'question':
                section_content = question
            elif topic_selector == 'narrative':
                section_content = narrative
            else:
                raise TypeError("Unsupported topic selector: %s" \
                        % topic_selector)

            yield Document(
                identifier=number,
                title=topic_type,
                sections=[Section(name=section, content=section_content)],
                collection='topics',
            )

def evaluate(corpus, topics, dictionary, index, filename, run_name, processor,
        settings_file):
    """
    Evaluates a topic model, producing a file that contains the collected
    relevance judgment predictions for each TREC topic.
    """

    with open(filename, 'w') as outfile:
        for topic in topics:
            topic_tokens = get_topic_tokens(topic, processor, settings_file)
            topic_doc = dictionary.doc2bow(topic_tokens)
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

def get_topic_tokens(topic, processor, settings_file):
    """
    Gets a list of tokens from a topic by processing it with the given
    processor. Currently, two processors are supported: one for using
    Parmenides and one for a simpler tokenizer/preprocessor using spaCy.
    """

    if processor == 'parmenides':
        get_parmenides_tokens(topic, settings_file)
    elif processor == 'spacy':
        get_spacy_tokens(topic)
    else:
        raise TypeError("Unsupported topic processor: %s" % processor)

def get_parmenides_tokens(topic, settings_file):

    if parmenides_processor is None:
        global parmenides_processor
        init(settings_file)
        parmenides_processor = import_class(settings.PROCESSOR)()

    return [str(term) for tree in parmenides_processor.process(topic) \
            for term in tree.terms]

def get_spacy_tokens(topic):

    if nlp is None:
        global nlp
        nlp = spacy.load('en')

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

if __name__ == "__main__":

    main()
