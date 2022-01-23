from bisturi.ontology import Ontology, Concept
from bisturi.model import Direction, ModuleID
from bisturi.util import filter_nan
from collections import Counter
from itertools import combinations
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from tqdm.auto import tqdm
from typing import Callable, Dict, List, Set, Tuple
import json
import numpy as np


Circuit = List[Tuple[Direction, Concept]]


def store_circuits(path: str, circuits: List[Circuit]) -> bool:
    circuits = [[(d, c.id) for d, c in circuit] for circuit in circuits]
    try:
        with open(path, 'w+') as fp:
            json.dump(circuits, fp, indent=2)
        return True
    except FileNotFoundError:
        return False


def load_circuits(path: str, ontology: Ontology) -> List[Circuit]:
    with open(path, 'r') as fp:
        circuits = json.load(fp)
        # Turn lists into tuples
        circuits = [[(((d[0][0], d[0][1]), d[1]), ontology.nodes[c])
                     for d, c in circuit] for circuit in circuits]
        return circuits


def retrieve_circuits(psi: Dict[ModuleID, Dict[int,
                                List[Tuple[Concept, float]]]],
                      semantic_check: Callable[Tuple[Concept, Concept], bool],
                      cache: str = None,
                      ontology: Ontology = None,
                      verbose: bool = False) -> List[Circuit]:
    '''
    Given a set of directions and their
    concepts it returns semantical
    consistent circuits.

    Parameters
    ----------
    psi: dict of list
        Dictionary mapping the module
        name to a list of (Concept,
        IOU) tuples for each direction
    semantic_check: Callable
        function to determine if
        two concept are semantically
        related
    verbose: bool, optional
        Prints to stdout the
        computation steps

    Returns
    -------
    circuits: list of circuits
        List containing for each
        circuit. Each circuit is
        composed by a list of its
        participants, expressed as
        a triple (module, direction_id, concept)
    '''

    # Try to read from file
    if cache and ontology:
        try:
            return load_circuits(cache, ontology)
        except FileNotFoundError:
            pass

    # List of (direction, concept) pairs
    pair_index = []

    # Keeps the index of the first
    # pair for each module
    start_module = []

    # Initialize the nodes of the graph
    idx = 0
    for module in psi:
        start_module.append(idx)
        for direction in psi[module]:
            concepts = psi[module][direction]
            for concept, sigma in concepts:
                pair_index.append(((module, direction), concept))
                idx += 1
    start_module.append(idx)
    n_pairs = len(pair_index)

    if verbose:
        print(n_pairs, 'total nodes')

    # Initialize the edges of the graph
    arcs = lil_matrix((n_pairs, n_pairs), dtype=np.int8)

    modules = list(psi.keys())

    # Progress bar
    total = sum([
        (start_module[i+1] - start_module[i])
        * (start_module[i+2] - start_module[i+1])
        for i, _ in enumerate(modules[:-1])])
    pbar = tqdm(total=total)

    for module_id, module in enumerate(modules[:-1]):
        # Iterate over the pair of directions in adjacent layers
        for a_node in pair_index[
                start_module[module_id]:start_module[module_id+1]]:
            a_idx = pair_index.index(a_node)
            for b_node in pair_index[
                    start_module[module_id+1]:start_module[module_id+2]]:
                # Unpack nodes into directions and concepts
                _, a_concept = a_node
                _, b_concept = b_node

                if semantic_check(a_concept, b_concept):
                    b_idx = pair_index.index(b_node)
                    arcs[a_idx, b_idx] = 1

                # Update progress
                pbar.update(1)

    # Terminate progress bar
    pbar.close()

    if verbose:
        print('Edges added')
        print('Computing connected components')

    n_components, labels = connected_components(csgraph=arcs,
                                                directed=False,
                                                return_labels=True)
    circuits = []
    for circuit in range(n_components):
        size = np.count_nonzero(labels == circuit)
        if size > 1:
            nodes = []
            for node in np.argwhere(labels == circuit):
                nodes.append(pair_index[int(node)])
            circuits.append(nodes)

    # Persistency
    if cache:
        store_circuits(cache, circuits)

    return circuits


def circuit_coherence(circuit: Circuit,
                      sim: Callable[Tuple[Concept, Concept], float],
                      unique: bool = False) -> float:
    '''
    Return the unique directions
    from a circuit.

    Parameters
    ----------
    circuit: list of tuples
        Circuit expressed as a list
        of tuples (module, direction_id, concept)
    sim: Callable
        Similarity function to determine
        the semantic similarity between
        two concepts
    unique: bool, optional
        If True it computes the similarity
        only between unique concepts.
        Defaults to false.

    Returns
    -------
    coherence: float
        Average of the similarities
        between circuit members.
    '''
    concepts = [c for d, c in circuit]

    if unique:
        concepts = set(concepts)

    if len(concepts) == 1:
        return 1.0

    coherence = np.average([sim(i, j) for i, j in combinations(concepts, r=2)])

    return coherence


def unique_directions(circuit: Circuit) -> Set[Direction]:
    return {d for d, _ in circuit}


def unique_concepts(circuit: Circuit) -> Set[Concept]:
    return {c for _, c in circuit}


def report_circuits(circuits: List[Circuit],
                    sim: Callable[Tuple[Concept, Concept], float]) -> Dict:
    return {
        'Circuits': len(circuits),
        'Average size': np.average([len(c) for c in circuits]),
        'Average directions': np.average(
            [len({d for d, _ in c}) for c in circuits]),
        'Average concepts': np.average(
            [len({concept for _, concept in c}) for c in circuits]),
        'Average coherency': np.average(
            [circuit_coherence(c, sim) for c in circuits]),
        'Average unique coherency': np.average(filter_nan(np.array(
            [circuit_coherence(c, sim, unique=True) for c in circuits]))),
    }


def smallest_dag(circuit: Circuit) -> List[Dict]:
    """
    Offers a visualization of the
    smallest connected DAG containing
    all the concepts within the circuit.
    """
    concepts = {c for _, c in circuit}
    ancestors = [c.get_ancestors() for c in concepts]
    mca = set.intersection(*ancestors)  # least common subsumer
    smallest_dag = set.union(*ancestors)
    for root in mca:
        smallest_dag = {c for c in smallest_dag if c in root.descendants}

    c_list = []
    for c in smallest_dag:
        hypernyms = set(c.hypernyms).intersection(smallest_dag)
        hyponyms = set(c.hyponyms).intersection(smallest_dag)
        c_list.append({'Concept': c.synset.name(),
                       'Aligned': c in concepts,
                       'Hypernyms': [h.synset.name() for h in hypernyms],
                       'Hyponyms': [h.synset.name() for h in hyponyms]})
    return c_list


def overview_circuit(circuit: Circuit,
                     sim: Callable[Tuple[Concept, Concept], float]) -> Dict:
    concepts = [c for _, c in circuit]
    directions = [d for d, _ in circuit]
    return {
        'Size': len(circuit),
        'Units': len(set(directions)),
        'Concepts': len(set(concepts)),
        'Hypernym': min(set(concepts),
                        key=lambda c: c.depth).synset,
        'Most Common': max(concepts, key=Counter(concepts).get),
        'Coherence': circuit_coherence(circuit, sim),
        'Unique coherence': circuit_coherence(circuit, sim, unique=True)
        }


def build_id(ss):
    return 10**8 + ss.offset()


def check_relatedness(a, b):
    '''
    a, b: Concepts

    Checks if a and b are
    related by an IsA or a PartOf
    semantical relation.
    '''
    a_cid, b_cid = a.id, b.id
    a_synset = a.synset
    b_lineage = b.lineage

    a_meronyms = {build_id(e) for e in a_synset.part_meronyms()}
    a_holonyms = {build_id(e) for e in a_synset.part_holonyms()}

    sim = int(bool(a_cid != b_cid and (a_cid in b_lineage or
                                       a_meronyms & b_lineage or
                                       a_holonyms & b_lineage)))

    return sim


def ssc(a, b, f, *p):
    '''
    Synset call.

    Calls the function f
    on a.synset and b.synset.
    '''
    return f(a.synset,
             b.synset, *p)


brown_ic = wordnet_ic.ic('ic-brown.dat')


sim_functions = {
    'jcn': lambda a, b: ssc(a, b, wn.jcn_similarity, brown_ic),
    'lch': lambda a, b: ssc(a, b, wn.lch_similarity),
    'lin': lambda a, b: ssc(a, b, wn.lin_similarity, brown_ic),
    'pth': lambda a, b: ssc(a, b, wn.path_similarity),
    'res': lambda a, b: ssc(a, b, wn.res_similarity, brown_ic),
    'wup': lambda a, b: ssc(a, b, wn.wup_similarity)
}
