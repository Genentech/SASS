"""Cutting a query molecule into fragment sets, and matching those sets to reactions.

`fragment_molecule` enumerates the ways a query can be split into `n` pieces. `FragNode` and
`FragNodeMatcher` decide whether a fragment set is compatible with a reaction's synthon
topology.
"""

# Standard Library
import hashlib
import logging
from itertools import permutations, product
from typing import Any, Callable, Iterator, Optional

# Third Party Library
from openeye import oechem

from sass.chem.mol import count_non_H_atom, get_conn_symbols, get_same_ring_bonds, substitute_atoms

logger = logging.getLogger(__name__)


def fragment_molecule(
    mol: oechem.OEMol,
    n_fragments: int,
    heavy_atom_limit: int,
    cleave_acyclic_bonds: bool = True,
    cleave_cyclic_bonds: bool = True,
    connector_isotope: int = 1,
    connector_atoms: list[str] | None = None,
    deduplicate: bool = True,
    count_halogens: bool = False,
) -> list[list[oechem.OEMol]]:
    """
    Recursively fragment a molecule into the desired number of fragments.

    Parameters
    ----------
    mol
        Input molecule to be fragmented.
    n_fragments
        Desired number of fragments.
    heavy_atom_limit
        Minimum number of heavy atoms to be considered a valid fragment.
    cleave_acyclic_bonds
        Whether to cleave acyclic bonds during fragmentation.
    cleave_cyclic_bonds
        Whether to cleave cyclic bonds during fragmentation.
    connector_atoms
        Atoms used to cap the cleaved bonds. By default, use the noble gases, since they
        are usually not found in organic reactants, and are different from the transuranium
        atoms used in Enamine synthon files.
    connector_isotope
        The isotope number of the connector atoms. Use a non-0 isotope number to not count
        connector atoms as heavy atoms when checking fragment sizes.
    deduplicate
        Whether to drop fragment sets that are identical once connector atoms are ignored.
        The recursion produces such duplicates. Connector assignments are enumerated later.
    count_halogens
        Whether to count halogens as heavy atoms. Default is False, since halogens are
        usually mono-valent and do not contribute to the connectivity of the molecule.

    Returns
    -------
    list[list[oechem.OEMol]]
        A list of lists of fragments.
    """

    # SMILES-based keys also deduplicate symmetric fragmentations.
    def get_frags_key(fragments: list[oechem.OEMol]) -> str:
        """
        "Hash" the fragments.

        Replace the connector atoms with C, generate SMILES, and sort and concat the strings.
        """
        nonlocal sub_dict

        smis = []
        for frag in fragments:
            _frag = oechem.OEMol(frag)
            substitute_atoms(_frag, sub_dict=sub_dict)
            smis.append(oechem.OEMolToSmiles(_frag))
        smis.sort()
        return "&".join(smis)

    def backtrack(fragments: list[oechem.OEMol], mol: oechem.OEMol, c_idx: int) -> None:
        """
        Recursively fragment the molecule.

        For all frags generated from `mol`, recurse into each frag, and append the rest
        of the frags to `fragments`. Deduplicate fragment sets in the end.

        Parameters
        ----------
        fragments
            List of fragments generated so far.
        mol
            Molecule to be fragmented.
        c_idx
            Index of the connector atom to be used for the next bond cleavage.
        """
        nonlocal frags_list, frags_key_set

        if len(fragments) == n_fragments - 1:
            res = [*fragments, mol]
            if not deduplicate:
                frags_list.append(res)
            else:
                frags_key = get_frags_key(res)
                if frags_key not in frags_key_set:
                    frags_key_set.add(frags_key)
                    frags_list.append(res)
            return

        if len(fragments) > n_fragments - 1:
            return

        # Stop search if the current mol to be split is too small.
        if check_fragment_size([mol], count_non_H_atom, heavy_atom_limit * 2, exclude_symbols=exclude_symbols) is False:
            return

        # Cut 1 acyclic bond, or 2 ring bonds
        oechem.OEFindRingAtomsAndBonds(mol)  # call this each time to re-assign ring membership

        # Store processed bond pairs to avoid duplicate processing.
        seen_ring_bond_pairs = set()

        for bond in mol.GetBonds():
            if cleave_acyclic_bonds and not bond.IsInRing():
                b_idx = bond.GetIdx()
                _mol = oechem.OEMol(mol)  # Do not cleave the original mol, otherwise hard to backtrack.
                _frags = get_n_fragments(
                    mol=_mol,
                    conn_atom_map={b_idx: (connector_atoms[c_idx], connector_isotope, bond.GetOrder())},
                )
                if len(_frags) == 2 and (
                    heavy_atom_limit is None
                    or check_fragment_size(
                        _frags,
                        count_non_H_atom,
                        # Empirically better than `count_non_ring_atom` for acyclic cuts.
                        heavy_atom_limit,
                        exclude_symbols=exclude_symbols,
                    )
                ):
                    for i, _ in enumerate(_frags):
                        backtrack(fragments + _frags[:i] + _frags[i + 1 :], _frags[i], c_idx + 1)
            elif cleave_cyclic_bonds and bond.IsInRing():
                # Restrict bond2 to only bonds in the same ring.
                same_ring_bonds = get_same_ring_bonds(bond, include_self=False)

                # Label the atoms in the ring with `ring_label`
                for b in same_ring_bonds:
                    for atom in (b.GetBgn(), b.GetEnd()):
                        atom.SetData(ring_label, True)

                for bond2 in same_ring_bonds:
                    b_idx1 = bond.GetIdx()
                    b_idx2 = bond2.GetIdx()
                    bond_pair = tuple(sorted([b_idx1, b_idx2]))
                    if bond_pair not in seen_ring_bond_pairs:
                        seen_ring_bond_pairs.add(bond_pair)
                        _mol = oechem.OEMol(mol)
                        _frags = get_n_fragments(
                            mol=_mol,
                            conn_atom_map={
                                b_idx1: (connector_atoms[c_idx], connector_isotope, bond.GetOrder()),
                                b_idx2: (connector_atoms[c_idx + 1], connector_isotope, bond2.GetOrder()),
                            },
                        )
                        # Still need to check, since being in the same ring doesn't guarantee
                        # generating 2 fragments (e.g. fused rings).
                        if len(_frags) == 2:  # noqa: SIM102
                            if heavy_atom_limit is None or check_fragment_size(
                                _frags,
                                count_non_ring_atom,
                                heavy_atom_limit,
                                ring_label=ring_label,
                                exclude_symbols=exclude_symbols,
                            ):
                                for i, _ in enumerate(_frags):
                                    backtrack(fragments + _frags[:i] + _frags[i + 1 :], _frags[i], c_idx + 2)

                # Unlabel the atoms in the ring for the next iteration.
                for b in same_ring_bonds:
                    for atom in (b.GetBgn(), b.GetEnd()):
                        atom.DeleteData(ring_label)

    if connector_atoms is None:
        connector_atoms = ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"]
    mol = oechem.OEMol(mol)  # Make a copy.
    ring_label = "prev_ring_atom"
    oechem.OEFindRingAtomsAndBonds(mol)

    # For deduplication
    sub_atom = "C"
    sub_isotope = 0
    sub_dict = {(conn, connector_isotope): (sub_atom, sub_isotope) for conn in connector_atoms}
    frags_key_set = set()

    frags_list = []
    exclude_symbols = set(connector_atoms)
    if not count_halogens:
        # Exclude halogens from the heavy atom count.
        exclude_symbols.update({"F", "Cl", "Br", "I", "At"})
    backtrack([], mol, 0)
    return frags_list


def get_n_fragments(
    mol: oechem.OEMol,
    conn_atom_map: dict[int, tuple[str, int, int]],
    conn_atom_3D: bool = True,
    clear_conn_atom_data: bool = True,
    connector_atom_tag: Optional[tuple[str, Any]] = None,
) -> list[oechem.OEMolBase]:
    """
    Split input molecule along the specified bonds and cap each cut with a connector atom.

    Bonds are given by index, since `mol` is often a copy of the molecule they were chosen on.

    Parameters
    ----------
    mol
        Molecule to be fragmented.
    conn_atom_map
        Mapping of bond idx to be cleaved to the type of connector atom
        to cap that bond with. Format: {b_idx: (<atom symbol>, <isotope>, <bond order)}.
    conn_atom_3D
        Whether to position the connector atom at the original position of the atom
        across the bond cleaved, or at the center of mass of molecule (default location).
    clear_conn_atom_data
        Whether to clear the name and tagged data the cap (dummy) atom copies from its
        template, so that only the original atom keeps its tags.
    connector_atom_tag
        Tag name and value to assign to the connector atoms.
    """
    for bond in mol.GetBonds():
        b_idx = bond.GetIdx()
        if b_idx in conn_atom_map:
            a1, a2 = bond.GetBgn(), bond.GetEnd()
            mol.DeleteBond(bond)
            conn_atom, isotope, bond_order = conn_atom_map[b_idx]

            for atom, template in ((a1, a2), (a2, a1)):
                if conn_atom_3D:
                    # Use the other atom as template for the dummy (to copy the position)
                    dummy = mol.NewAtom(template)
                else:
                    # Create a new atom at the center of mass.
                    dummy = mol.NewAtom(oechem.OEGetAtomicNum(conn_atom))

                if clear_conn_atom_data:
                    dummy.SetName("")
                    dummy.Clear()

                dummy.SetAtomicNum(oechem.OEGetAtomicNum(conn_atom))
                dummy.SetIsotope(isotope)
                mol.NewBond(atom, dummy, order=bond_order)
                if connector_atom_tag is not None:
                    key, val = connector_atom_tag
                    dummy.SetData(key, val)

    # Split `mol` into components
    return list(generate_components(mol))


def generate_components(mol: oechem.OEMol) -> Iterator[oechem.OEMol]:
    """
    Splits components of a multi-component molecule into separate molecules.

    Adapted from https://docs.eyesopen.com/toolkits/python/_downloads/9d7132664fe135a989141fdc9aef0af7/parts2mols.py
    """
    numparts, partlist = oechem.OEDetermineComponents(mol)
    pred = oechem.OEPartPredAtom(partlist)
    for i in range(1, numparts + 1):
        pred.SelectPart(i)
        partmol = oechem.OEGraphMol()
        oechem.OESubsetMol(partmol, mol, pred)
        yield partmol


def check_fragment_size(
    fragments: list[oechem.OEMol],
    filter_fn: Callable,
    heavy_atom_count: int,
    **kwargs: str,
):
    """
    Check whether all fragments pass the heavy-atom filter.

    Parameters
    ----------
    fragments
        An array of mol objects. E.g. generated from fragmenting a parent mol.
    filter_fn
        Function used to count the size (heavy-atom) of each fragment.
    heavy_atom_count
        The minimum size of fragments allowed.
    kwargs
        Keyword args for the `filter_fn`.
    """
    return all(filter_fn(frag, **kwargs) >= heavy_atom_count for frag in fragments)


def count_non_ring_atom(mol: oechem.OEMol, ring_label: str, exclude_symbols: set[str] = set()) -> int:
    """
    Count the number of heavy atoms that are not part of the ring-fragment.

    This is not the same as counting atoms where `IsInRing()` is false. Ring-fragment atoms
    must be labeled with `ring_label` beforehand. Atoms in `exclude_symbols` are not counted.

    Parameters
    ----------
    mol
        Input molecule.
    ring_label
        Text label designating an atom in ring-fragments.
    exclude_symbols
        Atoms matching any of these symbols will not be included in the count.
    """
    count = 0
    for atom in mol.GetAtoms():
        a_num = atom.GetAtomicNum()
        if a_num != 1 and oechem.OEGetAtomicSymbol(a_num) not in exclude_symbols and not atom.HasData(ring_label):
            count += 1

    return count


class FragNode:
    """Graph nodes representing molecule fragments.

    Fragment connectivity is indicated by the connector atoms. Each pair of connected
    fragments shares an edge defined by the common connector atoms. For example, for
    two fragments that form a ring, even though there are two connector atoms, they still
    only have one common edge, but the edge is labeled with two connector atoms.
    """

    def __init__(
        self,
        index: Any,
        smiles: str,
        conn_symbols: set[str],
    ) -> None:
        """Instantiate a FragNode object.

        Parameters
        ----------
        index
            Index of the nodes. Often the index of the node in a list of nodes.
        smiles
            SMILES of the fragment.
        conn_symbols
            Connector atoms on this fragment.
        """
        self.index = index
        self.smiles = smiles
        self.conn_symbols = conn_symbols
        self.neis: list = []  # need list to ensure order.
        self.nei_edges: dict = {}

    def add_nei(self, nei: "FragNode", common_conns: set[str]):
        self.neis.append(nei)
        self.nei_edges.update({nei: common_conns})

    def get_neis(self) -> list["FragNode"]:
        return self.neis

    def get_conns(self, nei: "FragNode") -> set[str]:
        return self.nei_edges[nei]


def map_mol_frag_graph(smiles: list[str], connector_set: set[str]) -> list[FragNode]:
    """Convert a list of SMILES into a graph of FragNodes.

    Parameters
    ----------
    smiles
        List of SMILES strings of the fragments.
    connector_set
        Set of all possible connector atoms in the fragments.

    Returns
    -------
        List of FragNode objects containing the edge information.
    """
    nodes = []
    for i, smi in enumerate(smiles):
        conn_symbols = set(get_conn_symbols(smi, connector_set))
        nodes.append(FragNode(i, smi, conn_symbols))

    for i, node in enumerate(nodes):
        for j, nei in enumerate(nodes):
            if i != j:
                common_conn = node.conn_symbols & nei.conn_symbols
                if len(common_conn) > 0:
                    node.add_nei(nei, common_conn)

    return nodes


class FragNodeMatcher:
    """Group of functions for enumerating ways to match two lists of fragments.

    A match is defined as a mapping of nodes from one graph to another, where each mapped
    nodes have the same number of neighbors and are visited at the same time during the
    traversal of two graphs.

    For example, for two graphs:
    {
        A: [B],
        B: [A, C],
        C: [B],
    }
    and
    {
        X: [Y],
        Y: [X, Z],
        Z: [Y],
    }

    A valid match is: {A: X, B: Y, C: Z}. Another valid match is: {A: Z, B: Y, C: X}.
    """

    @staticmethod
    def list_to_dict_graph(graph: list[FragNode]) -> dict[str | int : list[str | int]]:
        """Create a dict representation of the graph replacing nodes with simple immutable objects.

        The dict form is cheap to copy when enumerating permutations of the graph.

        Parameters
        ----------
        graph
            List of FragNode objects.

        Returns
        -------
            A dict representation of the graph, and a table to map immutable index to the node object.
        """
        table = {}
        out = {}
        for node in graph:
            if node.index not in table:
                table[node.index] = node
            out[node.index] = [nei.index for nei in node.neis]
        return out, table

    @staticmethod
    def permute_graph(graph: dict) -> list[dict]:
        """Permute the dict representation of the graph.

        Generate all combinations of different orders of the neighbors list of each node.
        This allows enumeration of all possible DFS traversal paths of the graph. There
        may be multiple permutations of the graph that lead to the same traversal path.

        Parameters
        ----------
        graph
            Graph in dict representation.

        Returns
        -------
            List of all possible permutations of the graph.
        """

        def _permute(idx: int | str, temp: dict) -> None:
            nonlocal res, nodes, graph
            if len(temp) == len(graph):
                res.append(dict(temp))  # a copy
                return
            node = nodes[idx]
            for neis in permutations(graph[node]):
                temp[node] = neis
                idx += 1
                _permute(idx, temp)
                idx -= 1
                del temp[node]

        res = []
        nodes = list(graph.keys())
        _permute(0, {})
        return res

    @staticmethod
    def match_nodes(graph1: dict, start1: Any, graph2: dict, start2: Any) -> dict:
        """Match nodes of the two graphs given two starting nodes.

        Parameters
        ----------
        graph1
            The reference graph.
        start1
            The starting node for graph1.
        graph2
            A particular permutation of the graph to match to the reference.
        start2
            The starting node for graph2.

        Returns
        -------
            A mapping of nodes from graph1 to graph2.
        """

        def dfs(node1: int | str, node2: int | str) -> None:
            nonlocal res, visited
            if node1 not in visited and node2 not in visited:
                visited.add(node1)
                visited.add(node2)
                if len(graph1[node1]) == len(graph2[node2]):
                    res[node1] = node2
                    for nei1, nei2 in zip(graph1[node1], graph2[node2], strict=False):
                        dfs(nei1, nei2)
                else:
                    return

        if len(graph1) != len(graph2):
            return None
        res = {}
        visited = set()
        dfs(start1, start2)
        if len(res) == len(graph1):
            return res

    @staticmethod
    def match_all_graphs(graph1: list[FragNode], graph2: list[FragNode]) -> list[dict]:
        """Generate all matches of two graphs. See class docstring for more details.

        Parameters
        ----------
        graph1:
            The reference graph in list format.
        graph2:
            The graph to match to the reference.

        Returns
        -------
            A list of all possible matches of the two graphs.
        """
        graph_dict1, map1 = FragNodeMatcher.list_to_dict_graph(graph1)

        # Modify the index of graph2 nodes to avoid conflict with graph1.
        for node in graph2:
            node.index = f"{node.index}_2"
        graph_dict2, map2 = FragNodeMatcher.list_to_dict_graph(graph2)

        all_graph_dict2 = FragNodeMatcher.permute_graph(graph_dict2)
        res = []
        start1 = next(iter(graph_dict1.keys()))  # Use any arbitrary node as a constant start on graph1.
        unique_matches = set()
        for g in all_graph_dict2:
            for start2 in g:  # Need to start from all nodes on graph2.
                matched = FragNodeMatcher.match_nodes(graph_dict1, start1, g, start2)
                if matched:
                    # Hash and deduplicate
                    hashed = FragNodeMatcher.hash_match(matched)
                    if hashed not in unique_matches:
                        unique_matches.add(hashed)
                        res.append(matched)

        # Convert the str/int index back to the original node object.
        node_matches = []
        for r in res:
            temp = {map1[key]: map2[val] for key, val in r.items()}
            node_matches.append(temp)

        return node_matches

    @staticmethod
    def hash_match(match_dict: dict) -> str:
        """'Hash' a match dictionary to a string for deduplication."""
        temp = [f"{key}&{val}" for key, val in match_dict.items()]
        temp.sort()
        return "$".join(temp)


def enumerate_fragment_connector_substitution(
    nodes1: list[FragNode],
    node_match: dict[FragNode, FragNode],
    cross_scoring: bool,
):
    """Enumerate all connector atom mapping given a node-matching scheme.

    A node matching is not sufficient to map the connector atoms, since there may be
    multiple connector atoms on each edge. For example, for two lists of fragments:

    [
        +----B  B----+
        |            |
    A---+----C, C----+
    ]
    and
    [
        +----Y  Y----+
        |            |
    X---+----Z, Z----+
    ]

    there are two nodes in each list (ignoring the fragment that connects to "A"), but
    because there are two connector atoms on the edge between two nodes, there are two
    valid mappings of the connector atoms: {B: Y, C: Z} and {B: Z, C: Y}.

    Parameters
    ----------
    nodes1
        List of nodes in the reference graph.
    node_match
        A valid match of the reference graph nodes to the other graph nodes.
    cross_scoring
        Whether to allow mapping of connector atoms on the reference and other graphs if
        the number of connector atoms on that edge are different. Only allowed for many-
        to-one or one-to-many mappings. For example, a the reference graph has two ring-
        forming fragments with 2 connector atoms (A, B) and the target graph has two non-
        ring-forming fragments with 1 connector atom (X), a valid mapping is
        {A: [X], B: [X]}. Another valid mapping is {X: [A, B]}.

    Returns
    -------
        A list of all possible connector atom mappings.
    """

    def order_conn_maps(conns1: list[str], conns2: list[str]) -> list[dict]:
        """
        Create combinations of connector atom mapping when there are more than 1 connectors.

        For example, for two lists of connectors [U, Np] and [He, Ar], there are 2 possible
        mappings: {U: He, Np: Ar} and {U: Ar, Np: He}.
        """
        res = []
        for _conns2 in permutations(conns2):
            temp = {c1: [c2] for c1, c2 in zip(conns1, _conns2, strict=False)}
            res.append(temp)
        return res

    def backtrack(idx: int, temp: dict) -> None:
        """Generate all connector atoms mappings based on node-matching scheme."""
        nonlocal res
        if idx == len(nodes1):
            res.append(dict(temp))
            return

        node1 = nodes1[idx]
        mappings = []
        # Collect mappings for connectors on all neighboring edges.
        for nei in nodes1[idx].neis:
            ref_conns = list(node1.nei_edges[nei])
            nei2 = node_match[nei]
            node2 = node_match[node1]
            dst_conns = list(node2.nei_edges[nei2])
            if not any(c in temp for c in dst_conns):  # Only add unseen connectors.
                if len(ref_conns) == len(dst_conns):
                    mappings.append(order_conn_maps(dst_conns, ref_conns))
                elif cross_scoring and any(len(x) == 1 for x in [ref_conns, dst_conns]):
                    mappings.append([dict.fromkeys(dst_conns, ref_conns)])

        # Add each of all combinations of mappings to the temp results.
        for mapping in product(*mappings):
            for m in mapping:
                for key in m:
                    temp[key] = m[key]
            backtrack(idx + 1, temp)
            for m in mapping:
                for key in m:
                    del temp[key]

    res = []
    backtrack(0, {})
    return res


def fragment_memo_key(mol: oechem.OEMolBase) -> tuple[str, str]:
    """Identify a query fragment by graph connectivity *and* 3D shape.

    `score_synthons` memoizes scoring on this key. The fragment is the ROCS reference, so two
    fragments may share scores only if they have the same shape. SMILES alone is not enough,
    since cutting two ring bonds can give one SMILES for two different parts of a ring.
    Element and position alone is not enough either, since it ignores bonds. Atoms are sorted,
    so the key does not depend on atom numbering.
    """
    coords = mol.GetCoords()
    packed = ";".join(
        sorted(
            f"{atom.GetAtomicNum()}:{coords[atom.GetIdx()][0]:.3f},"
            f"{coords[atom.GetIdx()][1]:.3f},{coords[atom.GetIdx()][2]:.3f}"
            for atom in mol.GetAtoms()
        )
    )
    return oechem.OEMolToSmiles(mol), hashlib.md5(packed.encode()).hexdigest()[:16]  # noqa: S324
