from nltk.corpus import wordnet as wn
from typing import Union, List, Generic, Dict

from bisturi.ontology import Concept, Ontology, T
from bisturi.util import synset_to_name


class WordnetConcept(Concept, Generic[T]):
    """
    Represents a concept in the
    Wordnet ontology.
    """
    def __init__(self,
                 synset: str,
                 hypernyms: Union[List[T], None] = None,
                 hyponyms: Union[List[T], None] = None):

        # Recover the WordNet IDs
        self.pos: str = synset[0]
        self.off: int = int(synset[1:])

        # Initialize concept
        super().__init__(self.off, synset, hypernyms, hyponyms)

    def is_placeholder(self) -> bool:
        return (len(self.hyponyms) == 1
                and self.hyponyms[0].leaves == self.leaves)

    @property
    def synset(self):
        return wn.synset_from_pos_and_offset(self.pos, self.off)

    def __repr__(self) -> str:
        return f'Concept(name={self.name}, synset={self.synset})'


class WordnetOntology(Ontology):
    def __init__(self,
                 nodes: Dict[str, WordnetConcept],
                 propagate: bool = True):

        # Construct hypernymy
        raw_ontology = []
        synsets = {c.synset for c in nodes.values()}
        while len(synsets) != 0:
            new_synsets = [ss.hypernyms() for ss in synsets]
            for hyponym, hypernyms in zip(synsets, new_synsets):
                for hypernym in hypernyms:
                    raw_ontology.append((synset_to_name(hypernym),
                                         synset_to_name(hyponym)))
            synsets = set(sum(new_synsets, []))
        raw_ontology = set(raw_ontology)

        # Construct the ontology
        hypernyms = set([e[0] for e in raw_ontology])
        hyponyms = set([e[1] for e in raw_ontology])
        synsets = hypernyms.union(hyponyms)
        root_syn = list(hypernyms - hyponyms)[0]

        # Higher-level concepts
        for synset in synsets:
            if synset not in nodes:
                nodes[synset] = WordnetConcept(synset)

        # Connect nodes
        for hypernym, hyponym in raw_ontology:
            nodes[hypernym].hyponyms += [nodes[hyponym]]
            nodes[hyponym].hypernyms += [nodes[hypernym]]

        # Identify root
        root = nodes[root_syn]

        # Cumulate Broden labels to eventually
        # retrieve higher level visual concepts
        if propagate:
            for synset in nodes:
                concept = nodes[synset]
                for descendant in concept.get_descendants():
                    concept.labels |= descendant.labels

        # Init superclass
        super().__init__(root)
