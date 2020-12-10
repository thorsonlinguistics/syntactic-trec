"""
Provides a Parmenides document source class for reading from NXML files (XML
files encoded using the NLM Journal Archiving and Interchange Tag Library).
Setting `DOCUMENT_SOURCE='syntrec.source.nxml.NXMLSource'` will enable
Parmenides to read documents directly from NXML files.
"""

import os
import warnings

from bs4 import BeautifulSoup
from parmenides.conf import settings
from parmenides.document import Document, Section
from parmenides.source import DocumentSource

class NXMLSource(DocumentSource):
    """
    A Parmenides document source for reading from NXML files. The input to the
    `get_documents` class method is a list of file paths, each of which is read
    from disk, parsed as XML, and converted into a Parmenides `Document`. All
    paragraphs are read from the file, but references and other metadata are
    ignored.
    """

    @classmethod
    def get_documents(cls, filenames):

        for filename in filenames:
            file_id = os.path.splitext(os.path.basename(filename))[0]

            try:
                with open(filename, 'r', errors=settings.ENCODING_ERRORS) \
                        as infile:

                    soup = BeautifulSoup(infile.read(), 'xml')

                    # Get article title (if it exists)
                    try:
                        article_title = \
                            ''.join(soup.front.find('article-meta')\
                                .find('title-group')\
                                .find('article-title')\
                                .strings
                            )
                    except AttributeError:
                        article_title = file_id

                    # Get sections
                    sections = []
                    try:
                        for sec in soup.body.find_all('sec', recursive=False):
                            sec_title = ''.join(sec.title.strings)
                            paragraphs = sec.find_all('p')
                            content = ''.join([''.join(paragraph.strings) for
                                paragraph in paragraphs])
                            sections.append(Section(name=sec_title,
                                content=content))
                    except AttributeError:
                        pass

                    yield Document(identifier=file_id,
                        title=article_title,
                        sections=sections,
                        collections=settings.COLLECTION_NAME,
                    )
            except FileNotFoundError:
                warnings.warn("Could not find file: %s" % filename)
