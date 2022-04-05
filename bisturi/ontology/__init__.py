from collections import deque
from typing import TypeVar, Union, List, Any, Generic, Set
from queue import Queue, LifoQueue

T = TypeVar('T', bound='Concept[Any]')


class Concept(Generic[T]):
    def __init__(self,
                 id: int,
                 name: str,
                 hypernyms: Union[List[T], None] = None,
                 hyponyms: Union[List[T], None] = None):

        # Attributes
        self.id: int = id
        self.name: str = name
        self.hypernyms: List[T] = hypernyms if hypernyms is not None else []
        self.hyponyms: List[T] = hyponyms if hyponyms is not None else []

        # Annotations in the dataset
        self.labels: Set = set()
        self.original_labels: Set = set()

        # Depth
        self.depth: int = 0

        # Cache calls
        self._cache_leaves = None
        self._cache_lineage = None
        self._cache_ancestors = None
        self._cache_descendants = None

    def is_leaf(self) -> bool:
        return not self.hyponyms

    def is_placeholder(self) -> bool:
        return (len(self.hyponyms) == 1
                and self.hyponyms[0].labels == self.labels)

    def is_propagated(self) -> bool:
        """
        Returns True if the concept
        is obtained via its children
        and does not have any original
        b_id associated
        """
        return len(self.original_labels) == 0

    def propagation_ratio(self) -> float:
        """
        Summarizes the gain in terms
        of associated b_ids given
        the children.

        Intuitively, the ratio is 0 if
        all the b_ids of the concept
        were manually assigned, while
        it is 1 if all of them are
        derived by the children of
        the ontology.
        """
        num = len(self.original_labels)
        dem = len(self.labels)
        return 1 - num / dem

    def get_leaves(self):
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

    def get_descendants(self):
        descendants = set()

        # Retrieve descendants
        frontier = deque([self])
        while frontier:
            node = frontier.pop()
            descendants.add(node)
            frontier.extend(node.hyponyms)

        return descendants

    def get_ancestors(self):
        ancestors = set()

        # Retrieve ancestors
        frontier = deque([self])
        while frontier:
            node = frontier.pop()
            ancestors.add(node)
            frontier.extend(node.hypernyms)

        return ancestors

    def get_lineage(self):
        '''
        Given a concept ID, it
        returns all of its
        ancestors and descendants.
        '''
        lineage = set()

        # Retrieve ancestors
        frontier = deque([self])
        while frontier:
            node = frontier.pop()
            lineage.add(node)
            frontier.extend(node.hypernyms)

        # Retrieve descendants
        frontier = deque([self])
        while frontier:
            node = frontier.pop()
            lineage.add(node)
            frontier.extend(node.hyponyms)

        return lineage

    def cache_leaves(self):
        self._cache_leaves = self.get_leaves()

    def cache_lineage(self):
        self._cache_lineage = self.get_lineage()

    def cache_ancestors(self):
        self._cache_ancestors = self.get_ancestors()

    def cache_descendants(self):
        self._cache_descendants = self.get_descendants()

    @property
    def leaves(self):
        if self._cache_leaves is None:
            self._cache_leaves = self.get_leaves()
        return self._cache_leaves

    @property
    def lineage(self):
        if self._cache_lineage is None:
            self._cache_lineage = self.get_lineage()
        return self._cache_lineage

    @property
    def descendants(self):
        if self._cache_descendants is None:
            self._cache_descendants = self.get_descendants()
        return self._cache_descendants

    @property
    def ancestors(self):
        if self._cache_ancestors is None:
            self._cache_ancestors = self.get_ancestors()
        return self._cache_ancestors

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'Concept(name={self.name})'


class Ontology(Generic[T]):
    def __init__(self, root: T):
        # Root node
        self.root: T = root

        # Retrieve nodes as list
        node_list = self.to_list(style='BFS', sort_by_id=False)

        # O(1) access to nodes
        self.nodes = {}
        for node in node_list:
            self.nodes[node.id] = node

        # Count nodes
        self.n: int = len(node_list)

        # Depth is computed as the minimum
        # distance from the root
        self.root.depth = 0

        # Having retrieved the nodes using a BFS visit
        # each node is positioned after at least one of
        # its parents. Moreover, they are the nearest
        # to the root.
        for node in node_list[1:]:
            parents_depths = [e.depth for e in node.hypernyms]
            node.depth = min(parents_depths) + 1

    def get_leaves(self):
        return self.root.get_leaves()

    @property
    def leaves(self):
        return self.root.leaves

    def __len__(self):
        return self.n

    def to_list(self, style='BFS', max_length=None,
                keep_placeholders=True, sort_by_id=True):
        node_list = []

        # Visit order
        if style == 'BFS':
            q = Queue()
        elif style == 'DFS':
            q = LifoQueue()
        else:
            raise ValueError(f'Unknown style: {style}')

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
