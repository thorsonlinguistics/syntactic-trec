"""
.. module:: trec.cord
    :platform: any
    :synopsis: allows Parmenides to process the CORD-19 dataset.

.. moduleauthor:: Jacob Collard <jacob@thorsonlinguistics.com>

This module provides a Parmenides document source for reading from the CORD-19
dataset.
"""

from parmenides.conf import settings
from parmenides.document import Document, Section
from parmenides.source import DocumentSource

import json
import os

class Cord19Source(DocumentSource):

    @classmethod
    def get_documents(cls, filenames):

        for filedesc in filenames:

            if filedesc.filename is None:
                yield Document(
                    identifier=filedesc.cord_uid,
                    title=filedesc.title,
                    sections=[Section(name='Abstract',
                        content=filedesc.abstract)],
                    collection=settings.COLLECTION_NAME
                )
            else:
                with open(filedesc.filename, 'r',
                        errors=settings.ENCODING_ERRORS) as infile:
                    data = json.load(infile)

                    sections = []

                    if 'abstract' in data['metadata'].keys():

                        paragraph_text = ' '.join([paragraph['text'] for paragraph in
                            data['metadata']['abstract']])

                        section = Section(name='Abstract', content=paragraph_text)
                        sections.append(section)

                    current_section = None
                    current_section_text = ''
                    for paragraph in data['body_text']:
                        if paragraph['section'] != current_section:
                            if current_section is not None:
                                sections.append(Section(name=current_section,
                                    content=current_section_text))
                            current_section = paragraph['section']
                            current_section_text = ''

                        current_section_text += (' %s' % paragraph['text'])\
                                .replace('q q', '')

                    sections.append(Section(name=current_section,
                        content=current_section_text))

                    yield Document(
                        identifier=filedesc.cord_uid,
                        title=filedesc.title,
                        sections=sections,
                        collection=settings.COLLECTION_NAME
                    )
