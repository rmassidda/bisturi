import unittest

from bisturi.ontology import Concept, Ontology


class ConceptTest(unittest.TestCase):
    def setUp(self):
        # Root
        self.root = Concept(0)

        # Children
        self.c_a = Concept(1)
        self.c_b = Concept(2)
        self.c_c = Concept(3)
        self.c_d = Concept(4)

        # Root -> C_a
        self.root.hyponyms.append(self.c_a)
        self.c_a.hypernyms.append(self.root)
        # Root -> C_b
        self.root.hyponyms.append(self.c_b)
        self.c_b.hypernyms.append(self.root)
        # C_b -> C_c
        self.c_b.hyponyms.append(self.c_c)
        self.c_c.hypernyms.append(self.c_b)
        # C_b -> C_d
        self.c_b.hyponyms.append(self.c_d)
        self.c_d.hypernyms.append(self.c_b)
        # Root -> C_d
        self.root.hyponyms.append(self.c_d)
        self.c_d.hypernyms.append(self.root)

        # Create ontology
        self.ontology = Ontology(self.root)

    def test_descendants(self):
        # Root
        self.assertEqual(self.root.descendants,
                         {self.root, self.c_a, self.c_b, self.c_c, self.c_d})

        # C_a
        self.assertEqual(self.c_a.descendants,
                         {self.c_a})

        # C_b
        self.assertEqual(self.c_b.descendants,
                         {self.c_b, self.c_c, self.c_d})

        # C_c
        self.assertEqual(self.c_c.descendants,
                         {self.c_c})

        # C_d
        self.assertEqual(self.c_d.descendants,
                         {self.c_d})

    def test_ancestors(self):
        # Root
        self.assertEqual(self.root.ancestors,
                         {self.root})

        # C_a
        self.assertEqual(self.c_a.ancestors,
                         {self.c_a, self.root})

        # C_b
        self.assertEqual(self.c_b.ancestors,
                         {self.c_b, self.root})

        # C_c
        self.assertEqual(self.c_c.ancestors,
                         {self.c_c, self.c_b, self.root})

        # C_d
        self.assertEqual(self.c_d.ancestors,
                         {self.c_d, self.c_b, self.root})

    def test_lineage(self):
        # Root
        self.assertEqual(self.root.lineage,
                         {self.root, self.c_a, self.c_b, self.c_c, self.c_d})

        # C_a
        self.assertEqual(self.c_a.lineage,
                         {self.c_a, self.root})

        # C_b
        self.assertEqual(self.c_b.lineage,
                         {self.c_b, self.c_c, self.c_d, self.root})

        # C_c
        self.assertEqual(self.c_c.lineage,
                         {self.c_c, self.c_b, self.root})

        # C_d
        self.assertEqual(self.c_d.lineage,
                         {self.c_d, self.c_b, self.root})

    def test_leaves(self):
        # Root
        self.assertEqual(self.root.leaves,
                         {self.c_a, self.c_c, self.c_d})

        # C_a
        self.assertEqual(self.c_a.leaves,
                         {self.c_a})

        # C_b
        self.assertEqual(self.c_b.leaves,
                         {self.c_c, self.c_d})

        # C_c
        self.assertEqual(self.c_c.leaves,
                         {self.c_c})

        # C_d
        self.assertEqual(self.c_d.leaves,
                         {self.c_d})

    def test_ontology(self):
        # BFS traversal
        self.assertEqual(
            self.ontology.to_list(style='BFS'),
            [
                self.root,
                self.c_a,
                self.c_b,
                self.c_c,
                self.c_d
            ]
            )

        # DFS traversal
        self.assertEqual(
            self.ontology.to_list(style='DFS'),
            [
                self.root,
                self.c_a,
                self.c_b,
                self.c_c,
                self.c_d
            ]
            )
