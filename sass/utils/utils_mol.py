"""Functions for molecule handling."""

# Standard Library
import logging
from collections import defaultdict, deque
from itertools import permutations, product
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Union

# Third Party Library
from openeye import oechem, oequacpac


def load_first_mol(
    file: Union[str, Path],
    clear_title: bool = True,
    mol_cleanup: bool = True,
    load_GraphMol: bool = False,
) -> oechem.OEMol:
    """Load the first molecule from a file.

    Parameters
    ----------
    file
        Path to the file containing molecules.
    clear_title
        Whether to clear the title of the molecule.
    mol_cleanup
        Whether to clean up the molecule.
    load_GraphMol
        Whether to load the molecule as a `GraphMol` instead of `OEMol`.    
    """
    try:
        if load_GraphMol:
            mol = oechem.OEGraphMol(next(oechem.oemolistream(str(file)).GetOEGraphMols()))
        else:
            mol = oechem.OEMol(next(oechem.oemolistream(str(file)).GetOEMols()))
        if clear_title:
            mol.SetTitle("")
        if mol_cleanup:
            cleanup_mol(mol)
    except StopIteration:
        logging.exception(f"No molecule found in {file}.")
    else:
        return mol


def smiles_to_mol(smiles: str, title: str = "") -> oechem.OEGraphMol:
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smiles)
    mol.SetTitle(str(title))
    return mol


def substitute_atoms(
    mol: oechem.OEMolBase,
    sub_dict: dict,
    delete_H: bool = False,
) -> oechem.OEMolBase:
    """
    Substitute atoms according to the input mapping.

    This function modifies the input mol AND returns the modified mol. The sub_dict
    should not contain any chain substitutions, e.g. C -> U, U -> Np, ...

    Parameters
    ----------
    mol
        Input molecule.
    sub_dict
        Mapping of which atom to substitute to what. Format:
        {(<src_atom_symbol>, <src_isotope>): (<dst_atom_sybmol>, <dst_isotope>)},
        e.g.: {('U', 0): ('C', 13), ('Xe', 22): ('Np', 0)}.
    delete_H
        Whether to delete the H atoms on the substituted atoms.

    Returns
    -------
    oechem.OEMolBase
        The modified molecule after atom substitution.

    """
    atom_subbed = False
    for a in mol.GetAtoms():
        a_symbol = oechem.OEGetAtomicSymbol(a.GetAtomicNum())
        a_isotope = a.GetIsotope()
        key = (a_symbol, a_isotope)
        if key in sub_dict:
            set_atom_type(a, *sub_dict[key])
            atom_subbed = True
            if delete_H:  # Is this difficult from OESuppressHydrogen?
                for nei in a.GetAtoms():
                    if nei.GetAtomicNum() == 1:
                        mol.DeleteAtom(nei)

    if atom_subbed is False:
        logging.warning("No atom was substituted.")
    return mol


def cleanup_smiles(smiles: str) -> oechem.OEMol:
    """Clean up input SMILES using `cleanup_mol`."""
    mol = smiles_to_mol(smiles)
    cleanup_mol(mol)
    return oechem.OEMolToSmiles(mol)


def implicit_hydrogen(mol: oechem.OEMolBase) -> oechem.OEMolBase:
    """
    Use `ImplicitHydrogens` method to clean up mol.

    Assign correct valence and formal charge, while allowing future enumeration
    of stereocenters. Modifies the input molecule in-place AND returns the mol.
    """
    oechem.OEAssignImplicitHydrogens(mol)
    oechem.OEAssignFormalCharges(mol)
    return mol


# TODO: expand to a more comprehensive filter/cleanup, with options for deep-clean.
def cleanup_mol(
    mol: oechem.OEMolBase,
    valence_method: Callable = oechem.OEAssignMDLHydrogens,
    protonation: bool = False,
) -> oechem.OEMol:
    """
    Clean up input molecule.

    Modifies input `mol` in-place AND returns the `mol`.

    Parameters
    ----------
    mol
        Input molecule.
    valence_method
        Method for completing the valence of all atoms.
    protonation
        Whether to return a protonated or neutral molecule.
    """
    valence_method(mol)

    # `OEGetReasonableProtomer` sets BOTH tautomer and charges. Remove charges if needed downstream.
    if not oequacpac.OEGetReasonableProtomer(mol):
        # This function protonates the `mol` in-place and gets a reasonable tautomer.
        logging.error(f"Unable to process mol {mol.GetTitle()}, SMILES: {oechem.OEMolToSmiles(mol)}.")
    if protonation is False:
        oequacpac.OERemoveFormalCharge(mol)

    oechem.OEAssignAromaticFlags(mol)

    return mol


def fragment_molecule(
    mol: oechem.OEMol,
    n_fragments: int,
    heavy_atom_limit: int,
    cleave_acyclic_bonds: bool = True,
    cleave_cyclic_bonds: bool = True,
    connector_isotope: int = 1,  # TODO: Deprecate.
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
        Whether to deduplicate the resulting fragment sets. Due to the need to recurse
        into each of the fragments from a split step, there are deduplicate fragment sets
        where the fragments are the same when not considering the connector atoms.
        Fragments are considered the same if they are the same after removing the connector
        atoms. The assignment of connector atoms in different orders is done in subsequent
        steps.
    count_halogens
        Whether to count halogens as heavy atoms. Default is False, since halogens are
        usually mono-valent and do not contribute to the connectivity of the molecule.

    Returns
    -------
    list[list[oechem.OEMol]]
        A list of lists of fragments.
    """

    # TODO: to replace with original bond indices as the frags_key.
    # On a second thought, "hashing" the SMILES allows deduplciation (in case of symmetrical molecules).
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
                        # Note: using `count_non_H_atom` could result in small fragments that contain below
                        # `heavy_atom_limit` number of non-ring atoms, i.e. previously generated ring-fragments
                        # undergoing further cleavage on a "linear" bond. However, experimentally this seem to
                        # give good results (better than using `count_non_ring_atom` filter).
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
    # ring_bond_label = "parent_ring_bond"
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
    Split input molecule along the specified bond.

    And cap the bond cleaved with desired atom type/isotope.

    It uses bond indices instead of bond object to specify which bonds to cleave, since
    often the input `mol` is a copy of the original mol to be cleaved, so the bonds are
    not the same objects as the bonds in the original mol, but the bond indices are the
    same in the copied `mol`.

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
        Whether to clear the label of the cap (dummy) atom, if copied from template.
        Also clears all tagged data. Necessary to avoid mixing up when selecting fragments
        based on tags (only the original atom should keep the tag, not the copied atom.)
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

    Copied from https://docs.eyesopen.com/toolkits/python/_downloads/9d7132664fe135a989141fdc9aef0af7/parts2mols.py
    """
    numparts, partlist = oechem.OEDetermineComponents(mol)
    pred = oechem.OEPartPredAtom(partlist)
    for i in range(1, numparts + 1):
        pred.SelectPart(i)
        partmol = oechem.OEGraphMol()
        oechem.OESubsetMol(partmol, mol, pred)
        yield partmol


def generate_heavy_isotope_atoms(mol: oechem.OEMol) -> list[oechem.OEAtomBase]:
    """Generator that yields atom pointers of non-standard isotopic atoms."""
    for atom in mol.GetAtoms():
        if atom.GetIsotope() != 0:
            yield atom


def count_non_H_atom(mol: oechem.OEMolBase, exclude_symbols: set[str] = set()) -> int:
    """Count the number non-H atoms, excluding those in the `exclude_symbols` set."""
    n = 0
    exclude_symbols.add("H")
    for a in mol.GetAtoms():
        a_num = a.GetAtomicNum()
        if oechem.OEGetAtomicSymbol(a_num) not in exclude_symbols:
            n += 1
    return n


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

    This is NOT simply counting atoms that are not .IsRing()! The ring atoms need to be
    labeled with `ring_label` prior to this step. Ring atoms are usually the atoms that
    are along the shortest path between two ring connectors.

    It also exclude atoms in the `exclude_symbols` set.

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


def get_conn_symbols(smi: str, connector_set: set[str]) -> list[str]:
    """Return a list of connector atoms in a given SMILES string based on string matching."""
    return [char for char in connector_set if char in smi]


def get_conn_symbols_freq(mol: oechem.OEMolBase, connector_set: set[str]) -> dict[str, int]:
    """Return the connector atom frequency in the input mol."""
    res = defaultdict(int)
    for a in mol.GetAtoms():
        a_symbol = oechem.OEGetAtomicSymbol(a.GetAtomicNum())
        if a_symbol in connector_set:
            res[a_symbol] += 1
    return res


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
        self.smiles = smiles  # need this??
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


# put inside class?
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

        This is done to facilitate enumerating all permutations of the graph in dict form,
        because it is inefficient to copy the graph based on FragNodes due to needing to
        circularly copy all unique nodes.

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
                    for nei1, nei2 in zip(graph1[node1], graph2[node2]):
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
        # 1 start from all start nodes on graph 2
        # 2 permute all graph2
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

        # Modify the index of graph2 nodes to avoid conflict with graph1. May not be needed.
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
            temp = {c1: [c2] for c1, c2 in zip(conns1, _conns2)}
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
                    mappings.append([{dst: ref_conns for dst in dst_conns}])

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


def special_substitute_atoms(mol: oechem.OEMol, sub_map: dict[str, list[str]], isotope: int = 0) -> oechem.OEMol:
    """
    Substitute connector atoms in the input molecule.

    In the substitution mapping, if two keys map to the same destination atom, the two
    source atoms are substituted and merged into a geometric centroid. If a key maps to
    multiple destination atoms, multiple instances of the destination atoms are placed at
    the coordinate of the source atom. This is done to ensure matching of the connector
    atoms during ROCS scoring.

    Parameters
    ----------
    mol
        Input molecule to be substituted.
    sub_map
        Mapping of substitution, e.g. {'He': ['U'], 'Ne': ['Np', 'Pu'], 'Ar': ['U']}.
    isotope
        The desired isotope number of the substituted atom.

    Returns
    -------
        The substituted molecule.
    """
    freq = defaultdict(int)  # Counting the duplicate sub_atoms.
    mol = oechem.OEMol(mol)

    # Step 1: Substitute atoms based on `sub_map`.
    for atom in mol.GetAtoms():
        atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
        if atom_symbol in sub_map:
            sub_atom = sub_map[atom_symbol][0]  # should have at least one target.
            set_atom_type(atom, sub_atom, isotope)
            freq[sub_atom] += 1
            if len(sub_map[atom_symbol]) > 1:
                # Place additional connector atoms (no bond needed), for cross-score.
                for i in range(1, len(sub_map[atom_symbol])):
                    sub_atom = sub_map[atom_symbol][i]
                    new_atom = mol.NewAtom(atom)
                    set_atom_type(new_atom, sub_atom, isotope)
                    freq[sub_atom] += 1

    # Step 2: Swap any duplicate sub_atom with centroids.
    for key, val in freq.items():
        if val >= 2:
            mol = replace_atoms_with_centroid(
                mol,
                atoms=[(key, isotope)] * val,
                centroid_isotope=isotope,
                centroid_symbol=key,
                delete_original_atoms=True,
            )

    return mol


def label_synthon_ring_connector(  # TODO: move to utils_query.
    synthon_smiles: dict[int, str],
    connector_atoms: list[str],
    smirks: str,
) -> dict[str, "bool"]:
    """
    Determine whether a synthon connector forms a ring bond, or an acyclic bond.

    Label the atoms connected to the connector atoms (nei atoms) by the connector symbol,
    react by SMIRKS rule, check the bond type of the newly formed bonds, and mark the
    connector atoms accordingly.

    For example, for a reaction of [U]-[*:1].[U]-[*:2]>>[*:1]-[*:2], the only connector
    "U" is not a ring connector, so the return is {"U": "False"}.
    For a reaction of ([U]-[*:1].[Np]-[*:2]-[*:3]=[Pu]).[Pu]=[*:4].[Np]-[*:5]=[*:6]-[U]>>
    [*:1]-[*:6]=[*:5]-[*:2]-[*:3]=[*:4], the return is {"U": "True", "Np": "True", "Pu": "False"}.

    Parameters
    ----------
    synthon_smiles
        SMILES or SMARTS of synthons labeled by their reaction position index in SMIRKS.
    connector_atoms
        All possible connector atom symbols on synthons.
    smirks
        Reaction SMIRKS to react the synthons.

    Returns
    -------
    dict
        A mapping of connector atom symbol to a boolean value indicating whether it forms a ring connector.
    """
    connector_atoms = set(connector_atoms)
    connector_nei_tag = "connected_to"
    connector_nei_dict = defaultdict(list)  # e.g. {'U': [atom1, atom2]}

    synthon_mols = {}
    for key, smi in synthon_smiles.items():
        mol = smiles_to_mol(smi)
        synthon_mols[key] = mol

    for mol in synthon_mols.values():  # `mol` objects are mutable
        for atom in mol.GetAtoms():
            atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
            if atom_symbol in connector_atoms:
                for _atom in find_connected_atoms(atom):
                    cur_data = []
                    if _atom.HasData(connector_nei_tag):
                        cur_data = _atom.GetData(connector_nei_tag)
                    _atom.SetData(connector_nei_tag, [*cur_data, atom_symbol])
                    # Set the data as a list, in case multiple connector atom share the same neighbor atom.
                    # Caution: a list object won't survive pickling, or read/write to file.

    # React and check whether the formed bonds are in a ring.
    libgen = oechem.OELibraryGen(smirks)
    for i, rxnt in synthon_mols.items():
        ret_code = libgen.SetStartingMaterial(rxnt, i)
        if ret_code != 1:
            raise ValueError(f"Failed to add synthon {rxnt} as {i}-th reactant.")
    prod = oechem.OEMol(next(libgen.GetProducts()))

    oechem.OEFindRingAtomsAndBonds(prod)
    for atom in prod.GetAtoms():
        if atom.HasData(connector_nei_tag):
            for connector_symbol in atom.GetData(connector_nei_tag):
                connector_nei_dict[connector_symbol].append(atom)

    res = {}
    for key, val in connector_nei_dict.items():
        atom1, atom2 = val  # One connector symbol can only correspond to one bond formed / two atoms.
        bond = mol.GetBond(atom1, atom2)
        res[key] = str(bond.IsInRing())  # Use str(bool) to match the frag_connector labels.

    return res


def get_shortest_path_atoms(
    mol: oechem.OEMolBase,
    start_atom_symbol: str,
    start_isotope: int,
    end_atom_symbol: str,
    end_isotope: int,
    include_start_end: bool = True,
) -> Optional[list[oechem.OEAtomBase]]:
    """
    Get the atoms along the shortest path between two atoms in a molecule using BFS.

    The returned path optionally includes the start and end atom.

    Parameters
    ----------
    mol
        Input molecule.
    start_atom_symbol
        Symbol of the atom on one terminus of the path.
    start_isotope
        Isotope of the atom on one terminus of the path.
    end_atom_symbol
        Symbol of the atom on the other terminus of the path.
    end_isotope
        Isotope of the atom on the other terminus of the path.
    include_start_end
        Whether to include the start and end atom in the return.
    """
    s_atoms = list(find_atoms(mol, start_atom_symbol, start_isotope))
    if len(s_atoms) == 0:
        raise ValueError(f"Start atom {start_isotope}{start_atom_symbol} is not found.")
    if len(s_atoms) > 1:
        raise ValueError(f"Multiple start atoms {start_isotope}{start_atom_symbol} are found. Likely ambiguity.")
    s_atom = s_atoms[0]
    # Currently not checking the count of `end_atom`. Just return the shortest path to any, if found.

    visited_atoms = {s_atom}  # add to `visited` when enqueuing, not when visiting
    q = deque([[s_atom]])  # queue contains a list of paths (list[Atom])
    while q:
        cur_path = q.popleft()
        cur_atom = cur_path[-1]
        for b in cur_atom.GetBonds():
            nei = b.GetNbr(cur_atom)
            if nei not in visited_atoms:
                visited_atoms.add(nei)
                new_path = [*cur_path, nei]
                nei_symbol = oechem.OEGetAtomicSymbol(nei.GetAtomicNum())
                nei_isotope = nei.GetIsotope()
                if nei_symbol == end_atom_symbol and nei_isotope == end_isotope:
                    if include_start_end is False:
                        for a in (s_atom, nei):
                            new_path.remove(a)
                    return new_path
                    # ensures that it doesn't find the start_atom as the end,
                    # if happend to be labeled the same atom-type and isotope.
                q.append(new_path)

    logging.error(f"Unable to find a path between specified start {start_isotope} and end {end_isotope} atoms.")
    return None


def get_same_ring_bonds(
    bond: oechem.OEBondBase,
    include_self: bool,
) -> list[oechem.OEBondBase]:
    """
    Get all bonds in the same ring as the input bond.

    Note: the mol containing the bond should already have ring atoms and bonds assigned
    with the `oechem.OEFindRingAtomsAndBonds` function.

    Parameters
    ----------
    bond
        Input bond object.
    include_self
        Whether to include the input bond in the output.

    Returns
    -------
        List of bonds in the same ring as the input bond.
    """
    visited_bonds = {bond}
    start_atom = bond.GetBgn()  # Arbitrarily choosing one of the two atoms as start.
    end_atom = bond.GetEnd()
    q = deque([[bond, start_atom]])  # Last element is the ending atom of the last bond.
    while q:
        cur_path = q.popleft()
        s_atom = cur_path.pop()
        for nei_bond in s_atom.GetBonds():
            if nei_bond.IsInRing() and nei_bond not in visited_bonds:
                visited_bonds.add(nei_bond)
                next_start = nei_bond.GetNbr(s_atom)
                if next_start == end_atom:
                    if include_self is False:
                        return [*cur_path, nei_bond][1:]
                    else:
                        return [*cur_path, nei_bond]
                q.append([*cur_path, nei_bond, next_start])
    return []


def set_atom_type(atom: oechem.OEAtomBase, target_symbol: str, target_isotope: int) -> None:
    """
    Set the atomic symbol and isotope of an atom.

    Modifies the atom in-place.

    Parameters
    ----------
    atom
        The atom object to be modified.
    target_symbol
        Symbol of the atom type to be set.
    target_isotope
        Isotope of the atom to be set.
    """
    atom.SetAtomicNum(oechem.OEGetAtomicNum(target_symbol))
    atom.SetIsotope(target_isotope)


def find_atoms(
    mol: oechem.OEMolBase,
    atom_symbol: str,
    isotope: int = 0,
) -> Iterator[oechem.OEAtomBase]:
    """Yield atoms in the input molecule that matches the atomic symbol and isotope number."""
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == oechem.OEGetAtomicNum(atom_symbol) and a.GetIsotope() == isotope:
            yield a


def find_connected_atoms(
    atom: oechem.OEAtomBase,
    return_H: bool = False,
) -> Iterator[oechem.OEAtomBase]:
    """Generator that yields atoms connected to the input atom (neighbors)."""
    for nei in atom.GetAtoms():
        if return_H or nei.GetAtomicNum() != 1:  # Optionally yield H atoms.
            yield nei


def extend_dummy_bond(
    mol: oechem.OEMol,
    dummy_atom_symbol: str,
    dummy_isotope: int,
    new_dummy_symbol: str = "Si",
    new_isotope: int = 33,
    distance: float = 3,
    explicit_bond: bool = False,
) -> oechem.OEMol:
    """
    Extend the dummy bond vector on each conformer.

    Place a second dummy atom along the line formed by the original dummy atom and the
    atom connected to that dummy atom. The new dummy bond vector is defined as the line
    formed by the two dummy atoms.

    Parameters
    ----------
    mol
        Input molecule.
    dummy_atom_symbol
        Symbol of the original dummy atom in `mol`.
    dummy_isotope
        Isotope number of the original dummy atom.
    new_dummy_symbol
        Symbol of the 2nd dummy atom to be placed.
    new_isotope
        Isotope number of the 2nd dummy atom.
    distance
        Directional distance from the original dummy atom to the new dummy atom. New
        dummy atom is placed alonged the line formed by the original dummy atom and its
        neighbor atom. Positive distance means the new dummy atom is placed in the same
        direction as the (neighbor atom -> original dummy), negative distance means the
        opposite direction.
    explicit_bond
        Whether to create an explicit bond between the original dummy atom and the new
        dummy atom. Having a bond or not has no effect on ROCS.

    Returns
    -------
    oechem.OEMol
        A new molecule with the extended dummy bond vector.
    """
    mol = oechem.OEMol(mol)
    new_atom = mol.NewAtom(oechem.OEGetAtomicNum(new_dummy_symbol))
    new_atom.SetIsotope(new_isotope)
    if explicit_bond:
        dummy_atom = next(find_atoms(mol, dummy_atom_symbol, dummy_isotope))
        # Strictly speaking, should add error catching here for when the dummy atom is not found.
        mol.NewBond(new_atom, dummy_atom, 1)
    new_mol = oechem.OEMol(mol)
    new_mol.DeleteConfs()

    for conf in mol.GetConfs():
        dummy_atom = next(find_atoms(conf, dummy_atom_symbol, dummy_isotope))
        connected_atom = next(find_connected_atoms(dummy_atom))
        dummy_coord = conf.GetCoords(dummy_atom)
        connected_coord = conf.GetCoords(connected_atom)
        dummy_bond_length = oechem.OEGetDistance(conf, dummy_atom, connected_atom)
        scaled_distance_factor = distance / dummy_bond_length
        new_coord = [b + scaled_distance_factor * (b - a) for a, b in zip(connected_coord, dummy_coord)]
        conf.SetCoords(new_atom, new_coord)
        new_mol.NewConf(conf)

    return new_mol


def unset_stereo(atom: oechem.OEAtomBase):
    """
    Remove the stereochemical information from an atom.

    Works only on 2D molecules. On 3D molecules, the stereochemistry is inferred from
    the 3D coordinates, and unsetting has no effect on the actual stereochemistry of the atom.
    """
    neis = list(atom.GetAtoms())

    ret_code = atom.SetStereo(neis, oechem.OEAtomStereo_Tetra, oechem.OEAtomStereo_Undefined)
    if ret_code is False:
        logging.warning("Unsetting stereocenter failed.")


def delete_atom(mol: oechem.OEMol, atom_symbol: str, isotope: int) -> oechem.OEMol:
    """Delete all atoms in a molecule that match the specified atom symbol and isotope."""
    mol = oechem.OEMol(mol)  # make a copy
    for a in find_atoms(mol, atom_symbol, isotope):
        oechem.OESuppressHydrogens(a)  # Make H implicit.
        mol.DeleteAtom(a)  # This will also remove the connected bonds.

    return mol


def set_conf_centroid_coord(
    conf: oechem.OEConfBase,
    atoms: Iterable[tuple[str, int]],
    centroid_atom: oechem.OEAtomBase,
) -> oechem.OEMol:
    """
    Set the centroid atom to be at the geometric center of input atoms on a conformer.

    Parameters
    ----------
    conf
        Input conformer.
    atoms
        List of atoms to calculate the centroid from.
    centroid_atom
        The atom to set the centroid coordinate to.
    """
    # here the copy of `conf` does not have the `atom_to_set` pointer!
    atoms = set(atoms)

    # Take average of the dummy atoms coordinates.
    x = y = z = 0
    n = 0
    for symbol, isotope in atoms:
        atoms = list(find_atoms(conf, symbol, isotope))
        for atom in atoms:
            _x, _y, _z = conf.GetCoords(atom)
            x += _x
            y += _y
            z += _z
            n += 1
    if n == 0:
        raise ValueError("Input conf does not contain any of the specified atom types.")
    if n == 1:
        logging.warning("Only one matching atom was found in total. Centroid set at the position of that atom.")
    new_coord = [coord / n for coord in (x, y, z)]
    conf.SetCoords(centroid_atom, new_coord)
    return conf


def replace_atoms_with_centroid(
    mol: oechem.OEMolBase,
    atoms: Iterable[tuple[str, int]],
    centroid_symbol: str,
    centroid_isotope: int = 0,
    delete_original_atoms: bool = True,
) -> oechem.OEMol:
    """
    Replace the connector atoms with centroids in each conformer of a multi-conformer mol.

    This function is compatible with arbitrary number of connector atoms, even though current
    synthons have at most 2 ring-connector atoms.
    """
    # Use an unlikely atom type as the temp centroid, in case the desired centroid type
    # is among the `dummies`.
    _symbol, _isotope = "Xe", 777
    if (_symbol, _isotope) in atoms:
        raise ValueError(f"{_symbol}_{_isotope} is of the same type as one of the existing connector atoms.")

    mol = oechem.OEMol(mol)
    new_atom = mol.NewAtom(oechem.OEGetAtomicNum(_symbol))
    new_atom.SetIsotope(_isotope)
    # Change the coordinate on each conformer of the multi-conformer molecule.
    new_mol = oechem.OEMol(mol)
    new_mol.DeleteConfs()
    for conf in mol.GetConfs():
        new_conf = set_conf_centroid_coord(conf, atoms, new_atom)
        new_mol.NewConf(new_conf)

    if delete_original_atoms:  # TODO: rewrite to loop through atoms just once.
        for symbol, isotope in atoms:
            new_mol = delete_atom(new_mol, symbol, isotope)

    # Change the temp centroid into the desired final centroid type. Atom type change
    # can be done on the mol object level instead on each conformer.
    substitute_atoms(new_mol, sub_dict={(_symbol, _isotope): (centroid_symbol, centroid_isotope)})

    return new_mol


def get_distance(mol1: oechem.OEMol, mol2: oechem.OEMol, symbol: str, isotope: int) -> float:
    """Calculate the distance between two atoms on two different molecules."""
    atom1 = next(find_atoms(mol1, symbol, isotope))
    atom2 = next(find_atoms(mol2, symbol, isotope))
    return oechem.OEGetDistance(mol1, atom1, mol2, atom2)
