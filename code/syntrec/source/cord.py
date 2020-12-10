"""
Provides a Parmenides document source class for reading from the CORD-19
dataset. Setting `DOCUMENT_SOURCE='syntrec.source.cord.CordSource'` will enable
Parmenides to read documents directly from CORD-19.
"""

from parmenides.conf import settings
from parmenides.document import Document, Section
from parmenides.source import DocumentSource

import json
import os

class CordSource(DocumentSource):
    """
    A Parmenides document source for reading from the CORD-19 dataset. The
    input to the `get_documents` class method is a list of file descriptions
    (see the `FileDescription` class) which describe how a particular article
    is represented in CORD-19. Some articles have abstracts only, while others
    may be represented by PDF or PMC JSON files; the file description tells
    Parmenides where to find the articles data. 
    """

    @classmethod
    def get_documents(cls, file_descriptions):

        for file_description in file_descriptions:
            if file_description.filename is None:
                # Abstract-Only Document
                yield Document(
                    identifier=file_description.cord_uid,
                    title=file_description.title,
                    sections=[Section(name='Abstract',
                        content=file_description.abstract)],
                    collection=settings.COLLECTION_NAME,
                )
            else:
                # Full-text document, loaded from filesystem
                with open(file_description.filename, 'r',
                        errors=settings.ENCODING_ERRORS) as infile:
                    data = json.load(infile)

                    sections = []

                    # Add the abstract
                    if 'abstract' in data['metadata'].keys():

                        paragraphs = [paragraph['text'] for paragraph in \
                                data['metadata']['abstract']]
                        joined_text = ' '.join(paragraphs)

                        section = Section(name='Abstract', content=joined_text)
                        sections.append(section)

                    current_section = None
                    current_section_text = ''
                    # Add sections from the article body
                    for paragraph in data['body_text']:
                        if paragraph['section'] != current_section:
                            if current_section is not None:
                                sections.append(Section(name=current_section,
                                    content=current_section_text))
                            current_section = paragraph['section']
                            current_section_text = ''

                        current_section_text += ' %s' % paragraph['text']

                    # Add the final section
                    sections.append(Section(name=current_section,
                        content=current_section_text))

                    yield Document(
                        identifier=file_description.cord_uid,
                        title=file_description.title,
                        sections=sections,
                        collection=settings.COLLECTION_NAME,
                    )
