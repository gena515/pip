from __future__ import absolute_import, division, unicode_literals

from genshi.core import (
    COMMENT,
    DOCTYPE,
    END,
    END_CDATA,
    END_NS,
    PI,
    START,
    START_CDATA,
    START_NS,
    TEXT,
    XML_NAMESPACE,
    QName,
)

from ..constants import voidElements, namespaces
from . import _base


class TreeWalker(_base.TreeWalker):
    def __iter__(self):
        # Buffer the events so we can pass in the following one
        previous = None
        for event in self.tree:
            if previous is not None:
                for token in self.tokens(previous, event):
                    yield token
            previous = event

        # Don't forget the final event!
        if previous is not None:
            for token in self.tokens(previous, None):
                yield token

    def tokens(self, event, next):
        kind, data, pos = event
        if kind == START:
            tag, attribs = data
            name = tag.localname
            namespace = tag.namespace
            converted_attribs = {}
            for k, v in attribs:
                if isinstance(k, QName):
                    converted_attribs[(k.namespace, k.localname)] = v
                else:
                    converted_attribs[(None, k)] = v

            if namespace == namespaces["html"] and name in voidElements:
                for token in self.emptyTag(namespace, name, converted_attribs,
                                           not next or next[0] != END
                                           or next[1] != tag):
                    yield token
            else:
                yield self.startTag(namespace, name, converted_attribs)

        elif kind == END:
            name = data.localname
            namespace = data.namespace
            if name not in voidElements:
                yield self.endTag(namespace, name)

        elif kind == COMMENT:
            yield self.comment(data)

        elif kind == TEXT:
            for token in self.text(data):
                yield token

        elif kind == DOCTYPE:
            yield self.doctype(*data)

        elif kind in (XML_NAMESPACE, DOCTYPE, START_NS, END_NS,
                      START_CDATA, END_CDATA, PI):
            pass

        else:
            yield self.unknown(kind)
