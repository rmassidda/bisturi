from __future__ import annotations
from collections import deque
from nltk.corpus import wordnet as wn
from queue import Queue, LifoQueue
from typing import List


class Concept:
    """
    Represents a concept within an arbitrary ontology.

    Attributes
    ----------
    id : int
        The ID of the concept.
    hypernyms : List[Concept]
        The hypernyms of the concept.
    hyponyms : List[Concept]
        The hyponyms of the concept.
    depth : int
        The depth of the concept.
    leaves : Set[Concept]
        The leaves of the concept.
    ancestors : Set[Concept]
        The ancestors of the concept.
    descendants : Set[Concept]
        The descendants of the concept.
    lineage : Set[int]
        The union of ancestors and descendants.

    Methods
    -------
    is_leaf()
        Returns True if the concept is a leaf.
    is_placeholder()
        Returns True if the concept is a placeholder.
    """
    def __init__(self, concept_id: int,
                 hypernyms: List[Concept] = None,
                 hyponyms: List[Concept] = None):
        """
        Initializes a Concept object.

        Parameters
        ----------
        concept_id : int
            The ID of the concept.
        hypernyms : List[Concept]
            The hypernyms of the concept.
        hyponyms : List[Concept]
            The hyponyms of the concept.
        """
        # Unique ID for the concept
        self.id = concept_id

        # Hypernyms
        self.hypernyms = hypernyms if hypernyms is not None else []

        # Hyponyms
        self.hyponyms = hyponyms if hyponyms is not None else []

        # NOTE: .depth and .propagated depend
        #       respectively on the root and on
        #       the dataset, so they should be
        #       handled outside of this class.
        #       This is a temporary solution.

        # Default depth
        self.depth = 0

        # "Original concept" or propagated
        self.propagated = False

        # Cache related concepts
        self._cache_leaves = None
        self._cache_ancestors = None
        self._cache_descendants = None
        self._cache_lineage = None

    @property
    def leaves(self):
        if self._cache_leaves is None:
            self._cache_leaves = self._get_leaves()
        return self._cache_leaves

    @property
    def descendants(self):
        if self._cache_descendants is None:
            self._cache_descendants = self._get_descendants()
        return self._cache_descendants

    @property
    def ancestors(self):
        if self._cache_ancestors is None:
            self._cache_ancestors = self._get_ancestors()
        return self._cache_ancestors

    @property
    def lineage(self):
        if self._cache_lineage is None:
            self._cache_lineage = self.ancestors.union(self.descendants)
        return self._cache_lineage

    def is_leaf(self):
        """
        Returns True if the concept is a leaf.

        Returns
        -------
        bool
            True if the concept is a leaf.
        """
        return not self.hyponyms

    def is_placeholder(self):
        """
        Returns True if the concept is a placeholder.

        Returns
        -------
        bool
            True if the concept is a placeholder.
        """
        return (len(self.hyponyms) == 1
                and self.hyponyms[0].leaves == self.leaves)

    def _get_leaves(self):
        leaves = set()
        frontier = deque()
        frontier.append(self)
        while frontier:
            node = frontier.pop()
            if node.is_leaf():
                leaves.add(node)
            else:
                frontier.extend(node.hyponyms)

        return leaves

    def _get_descendants(self):
        descendants = set()

        # Retrieve descendants
        frontier = deque([self])
        while frontier:
            node = frontier.pop()
            descendants.add(node)
            frontier.extend(node.hyponyms)

        return descendants

    def _get_ancestors(self):
        ancestors = set()

        # Retrieve ancestors
        frontier = deque([self])
        while frontier:
            node = frontier.pop()
            ancestors.add(node)
            frontier.extend(node.hypernyms)

        return ancestors

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return f'Concept(id={self.id})'


class WordNetConcept(Concept):
    """
    Represents a concept within the WordNet ontology.

    Attributes
    ----------
    pos: str
        The part of speech of the concept.
    offset: int
        The offset of the concept.
    synset: Synset
        Synset in the WordNet ontology.
    wordnet_id: str
        The ID of the concept in WordNet format.
    """
    def __init__(self, wordnet_id: str,
                 hypernyms: List[WordNetConcept] = None,
                 hyponyms: List[WordNetConcept] = None):
        # WordNet ID
        self.wordnet_id = wordnet_id

        # Part of speech
        self.pos = self.wordnet_id[0]
        if self.pos != 'n':
            raise ValueError(f'WordNet only supports nouns, '
                             f'not {self.pos}')

        # Offset
        self.offset = int(self.wordnet_id[1:])

        # Initialize concept
        super().__init__(self.offset, hypernyms, hyponyms)

    @property
    def synset(self):
        # WordNet synset
        return wn.synset_from_pos_and_offset(self.pos, self.offset)

    def __str__(self):
        return str(self.synset)

    def __repr__(self):
        return f'Concept(id={self.id}, wordnet_id={self.wordnet_id}, '\
               f'synset={self.synset})'


class Ontology:
    """
    Represents an arbitrary ontology as
    a directed acyclic graph.
    """
    def __init__(self, root):
        # Root node
        self.root = root

        # Retrieve nodes as list
        node_list = self.to_list(style='BFS', sort_by_id=False,
                                 keep_placeholders=True)

        # Directly accessible concepts
        self.nodes = {}
        for node in node_list:
            self.nodes[node.id] = node

        # Count nodes
        self.n = len(node_list)

        # Having retrieved the nodes using a BFS visit
        # each node is positioned after at least one of
        # its parents. Moreover, they are the nearest
        # to the root.
        for node in node_list[1:]:
            parents_depths = [e.depth for e in node.hypernyms
                              if e.depth is not None]
            node.depth = min(parents_depths) + 1

    @property
    def leaves(self):
        return self.root.leaves

    def to_list(self, style='BFS', max_length=None,
                keep_placeholders=False, sort_by_id=True):
        """
        Returns a list of concepts in the ontology.

        Parameters
        ----------
        style : str
            The style of the traversal as in BFS or DFS.
        max_length : int
            The maximum length of the list.
        keep_placeholders : bool
            Whether to keep placeholders nodes.
        sort_by_id : bool
            Whether to sort the list by node ID.

        Returns
        -------
        node_list : List[Concept]
            The list of concepts.
        """

        # Empty list
        node_list = []

        # Visit order
        if style == 'BFS':
            q = Queue()
        elif style == 'DFS':
            q = LifoQueue()

        # Start visit
        q.put(self.root)
        while not q.empty():
            # Check list length
            if max_length and len(node_list) == max_length:
                return node_list

            # Retrieve node
            node = q.get()

            # Avoid duplicates
            if node not in node_list:
                node_list.append(node)

                # Iterate children
                for child in node.hyponyms:
                    q.put(child)

        if not keep_placeholders:
            node_list = [c for c in node_list if not c.is_placeholder()]

        if sort_by_id:
            node_list = sorted(node_list, key=lambda c: c.id)

        return node_list

    def __len__(self):
        return self.n
