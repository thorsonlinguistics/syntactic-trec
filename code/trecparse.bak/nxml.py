"""
.. module:: trec.nxml
    :platform: any
    :synopsis: allows Parmenides to process meaningful content from NXML files.

.. moduleauthor:: Jacob Collard <jacob@thorsonlinguistics.com>

This module provides a Parmenides document source for reading TREC NXML files.
"""

from bs4 import BeautifulSoup
from parmenides.conf import settings
from parmenides.document import Document, Section
from parmenides.source import DocumentSource

import os

class NXMLSource(DocumentSource):

    @classmethod
    def get_documents(cls, filenames):

        for filename in filenames:
            file_id = os.path.splitext(os.path.basename(filename))[0]
           
            try:
                with open(filename, 'r', errors=settings.ENCODING_ERRORS) \
                        as infile:

                    soup = BeautifulSoup(infile.read(), 'xml')

                    try:
                        article_title = \
                            ''.join(soup.front.find('article-meta').find('title-group').find('article-title').strings)
                    except AttributeError:
                        article_title = file_id
                    sections = []

                    try:
                        for sec in soup.body.find_all('sec', recursive=False):
                            sec_title = ''.join(sec.title.strings)
                            paragraphs = sec.find_all('p')
                            content = ''.join([''.join(paragraph.strings) for
                                paragraph in paragraphs])
                            sections.append(Section(name=sec_title, content=content))
                    except AttributeError:
                        pass
                    
                    yield Document(identifier=file_id, 
                            title=article_title, sections=sections,
                            collection=settings.COLLECTION_NAME)
            except FileNotFoundError:
                pass
