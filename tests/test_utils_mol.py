"""Test file for sass.utils.utils_mol module."""

import json
import math
import sys
from collections import defaultdict
from itertools import permutations

import pytest
import pytest_check as check
from openeye import oechem

from sass import TEST_DIR
from sass.utils.utils_mol import (
    FragNode,
    FragNodeMatcher,
    enumerate_fragment_connector_substitution,
    extend_dummy_bond,
    find_atoms,
    fragment_molecule,
    generate_components,
    get_conn_symbols_freq,
    get_same_ring_bonds,
    get_shortest_path_atoms,
    label_synthon_ring_connector,
    load_first_mol,
    map_mol_frag_graph,
    replace_atoms_with_centroid,
    special_substitute_atoms,
    substitute_atoms,
)

# from sass.utils.utils_data import SynthonHandler

DATA_DIR = TEST_DIR / "data"


@pytest.fixture()
def mc_mol_2conn():
    """Get a multi-conformer molecule with 2 connector atoms (U, Np)."""
    return next(oechem.oemolistream(str(DATA_DIR / "5349414_U_Np_50_conf.oez")).GetOEMols())


def get_atom_coordinate(conf, atom_symbol: str, atom_isotope: int) -> tuple[float, float, float]:
    atom = next(find_atoms(conf, atom_symbol, atom_isotope))  # assume only one match
    return conf.GetCoords(atom)


def test_replace_with_centroids(mc_mol_2conn: oechem.OEMol):
    # Manually calculate the coordinates of the centroids for each conformer.
    # The input molecule is know to have "U" and "Np" as the connectors.
    mol = mc_mol_2conn
    manual_centroid_coords = []
    for conf in mol.GetConfs():
        coord_U = get_atom_coordinate(conf, "U", 0)
        coord_Np = get_atom_coordinate(conf, "Np", 0)
        manual_coord = [(a + b) / 2 for a, b in zip(coord_Np, coord_U)]
        manual_centroid_coords.append(manual_coord)

    # Use the function to place centroids
    new_symbol, new_isotope = "Se", 333
    new_mol = replace_atoms_with_centroid(
        mol=mol,
        atoms=[("U", 0), ("Np", 0)],
        centroid_symbol=new_symbol,
        centroid_isotope=new_isotope,
        delete_original_atoms=True,
    )

    # Check that original connector atoms have been deleted.
    for atom in new_mol.GetAtoms():
        atom_symbol = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
        check.is_not_in(atom_symbol, ("U", "Np"))

    # Get the coordinates from the new_mol
    fn_centroid_coords = [get_atom_coordinate(conf, new_symbol, new_isotope) for conf in new_mol.GetConfs()]

    # Check if the coordinates from the fn is the same as manually calculated.
    for man_coord, fn_coord in zip(manual_centroid_coords, fn_centroid_coords):
        check.almost_equal(man_coord, fn_coord, abs=1e-6)


@pytest.mark.parametrize(
    "smi_in",
    [
        "CCCCCCC([U])=O",
        "O=C1N(C2CCC2)C3=CC(C(NC4=CC=C(C)N=C4)=O)=CC(Cl)=C3N1C5C[C@H](O)CC5",
        "O=S(CC1=NC(C2=CC=C(C)N=C2)=C(N3C[C@H](C(NC4CCC4)=O)CC3)O1)(NC(C)C)=O",
        "O=C1OC([C@H]([C@@H](C(C)C)Br)F)=N[C@@H]1N(C(C2)=O)CC[C@@H]2C3=C(C=CN4)C4=CC=C3",
    ],
)
@pytest.mark.parametrize(
    "sub_dict",
    [
        {("U", 0): ("Np", 0), ("Cl", 0): ("F", 0)},
        {("O", 0): ("F", 0), ("Br", 0): ("C", 0)},
        {("N", 0): ("O", 0), ("C", 0): ("P", 0), ("Br", 0): ("I", 0)},
    ],
)
def test_substitute_atoms(smi_in, sub_dict):
    """Source atoms are replaced by target atoms."""
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smi_in)
    sub_mol = oechem.OEMol(mol)
    substitute_atoms(sub_mol, sub_dict, delete_H=True)

    for (src_atom, _), (dst_atom, _) in sub_dict.items():
        if any(oechem.OEGetAtomicSymbol(a.GetAtomicNum()) == src_atom for a in mol.GetAtoms()):
            check.is_false(any(oechem.OEGetAtomicSymbol(a.GetAtomicNum()) == src_atom for a in sub_mol.GetAtoms()))
            check.is_true(any(oechem.OEGetAtomicSymbol(a.GetAtomicNum()) == dst_atom for a in sub_mol.GetAtoms()))


@pytest.mark.parametrize(
    ("mol", "n_frag", "ha_limit", "expected_count"),
    [
        (load_first_mol(DATA_DIR / "test_query_4.sdf"), 3, 5, 34),
        (load_first_mol(DATA_DIR / "test_query_4.sdf"), 2, 5, 17),
        (load_first_mol(DATA_DIR / "test_query_4.sdf"), 3, 4, 62),
        (load_first_mol(DATA_DIR / "test_query_4.sdf"), 3, 3, 194),
        (load_first_mol(DATA_DIR / "test_query_4.sdf"), 4, 5, 5),
    ],
)
def test_fragment_molecule(mol: oechem.OEMol, n_frag: int, ha_limit: int, expected_count):
    """Correct number of fragment sets are generated, and labeled with connectors."""
    frag_conns = ["U", "Np", "Pu", "Am", "Bk", "Ds", "Rg"]
    conn_set = set(frag_conns)
    frags = fragment_molecule(
        mol,
        n_fragments=n_frag,
        heavy_atom_limit=ha_limit,
        connector_atoms=frag_conns,
        count_halogens=True,
    )
    check.equal(len(frags), expected_count)
    for frag_set in frags:
        # Each fragment set contains `n_frag` fragments.
        check.equal(len(frag_set), n_frag)

        # Exactly 2 of each used connectors are in the frags.
        expected_freq = 2
        conn_freq = defaultdict(int)
        for frag in frag_set:
            _conn_freq = get_conn_symbols_freq(frag, conn_set)
            for key, val in _conn_freq.items():
                conn_freq[key] += val
        check.is_true(all(val == expected_freq for val in conn_freq.values()))

        # The connector atoms used are exacty the first n in the `frag_conns` list.
        check.is_true(all(key in frag_conns[: len(conn_freq)] for key in conn_freq))


@pytest.mark.parametrize(
    "smi_in",
    [
        "O=S(N1CCN(CC1)C)=O.CCCc2nn(C)c3c2N=CNC3=O.CCOc4ccccc4",
        "O=S(N1CCN(CC1)C)=O.Cn2c3c(N=CNC3=O)cn2.CCOc4ccccc4.CCC",
        "OC1=CC(C[14C])=C([13C])C=C1.[13C][C@]2([H])[C@@]([C@@](CC[C@@H]3O)([H])[C@]3(C)CC2)([H])C[14C]",
        "Fc1ccc(C2C(CNCC2)COc3ccc4c(OCO4)c3)cc1",
    ],
)
def test_generate_components(smi_in):
    """Correct number of components are generated from input molecules."""
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smi_in)
    components = list(generate_components(mol))
    expected_n = smi_in.count(".") + 1  # Components are separated by '.' in SMILES.
    check.equal(len(components), expected_n)


@pytest.mark.parametrize(
    ("smiles", "expected_connectivity"),
    [
        (
            ["[U]CC[Np]", "[U]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN", "[Pu]CC(CC[Bk])[Np]", "[Bk]CO"],
            {
                0: {1: {"U"}, 3: {"Np"}},
                1: {0: {"U"}, 2: {"Am"}, 3: {"Pu"}},
                2: {1: {"Am"}},
                3: {0: {"Np"}, 1: {"Pu"}, 4: {"Bk"}},
                4: {3: {"Bk"}},
            },
        ),
        (
            ["[U]CC[Pu]", "[U]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN"],
            {
                0: {1: {"U", "Pu"}},
                1: {0: {"U", "Pu"}, 2: {"Am"}},
                2: {1: {"Am"}},
            },
        ),
    ],
)
@pytest.mark.parametrize("connectors", [{"U", "Np", "Pu", "Am", "Bk"}])
def test_map_mol_frag_graph(smiles, connectors: list[str], expected_connectivity: dict):
    """Fragments are mapped to graphs with correct number of neighbors and correct connector atoms to neis."""
    nodes = map_mol_frag_graph(smiles, connectors)
    print(len(nodes))
    for node in nodes:
        print(node.index, len(node.neis))
        check.equal(len(node.neis), len(expected_connectivity[node.index]))
        for nei in node.neis:
            nei_conn = node.nei_edges[nei]
            check.equal(nei_conn, expected_connectivity[node.index][nei.index])


@pytest.mark.parametrize(
    "smiles",
    [
        ["[U]CC[Np]", "[U]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN", "[Pu]CC(CC[Bk])[Np]", "[Bk]CO"],
        ["[U]CC[Pu]", "[U]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN"],
    ],
)
@pytest.mark.parametrize("connectors", [{"U", "Np", "Pu", "Am", "Bk"}])
def test_list_to_dict_graph(smiles, connectors):
    """Graph is correctly converted from list to dictionary representation."""
    nodes = map_mol_frag_graph(smiles, connectors)
    graph_dict, mapping = FragNodeMatcher.list_to_dict_graph(nodes)
    for key, val in graph_dict.items():
        check.equal(key, mapping[key].index)
        check.equal(set(val), {nei.index for nei in mapping[key].neis})


@pytest.mark.parametrize(
    "graph",
    [
        {"a": ["b", "e"], "b": ["a", "c", "d"], "c": ["b"], "d": ["b"], "e": ["a"]},
        {
            "a": ["b", "d", "e"],
            "b": ["a", "c"],
            "c": ["b", "d", "f"],
            "f": ["c"],
            "d": ["c", "a"],
            "e": ["a"],
        },
        {
            "a": ["b"],
            "b": ["a", "c", "d"],
            "c": ["b", "d"],
            "d": ["c", "b", "e"],
            "e": ["d"],
        },
    ],
)
def test_permute_graph(graph):
    """Correct number of permutations are generated for a graph."""
    all_graphs = FragNodeMatcher.permute_graph(graph)
    # Calculate expected number of permutations
    n_perm = math.prod([len(list(permutations(graph[node]))) for node in graph])
    check.equal(len(all_graphs), n_perm)


@pytest.mark.parametrize(
    "graph",
    [
        {"a": ["b", "e"], "b": ["a", "c", "d"], "c": ["b"], "d": ["b"], "e": ["a"]},
        {
            "a": ["b", "d", "e"],
            "b": ["a", "c"],
            "c": ["b", "d", "f"],
            "f": ["c"],
            "d": ["c", "a"],
            "e": ["a"],
        },
        {
            "a": ["b"],
            "b": ["a", "c", "d"],
            "c": ["b", "d"],
            "d": ["c", "b", "e"],
            "e": ["d"],
        },
    ],
)
def test_match_nodes(graph):
    """Nodes are matched successfully or fail to match based on the starting node.

    In this test, a duplicate of the input graph with modified node names but same order
    of the neighbors list is used to match to the input graph. It is expected that there
    is one match starting from the same node as the input graph, and no matches starting
    from other nodes.
    For the matched graphs, since the test graph has the same order of neighbors as the
    input, the match dict should have the key an values being the same (with modified
    node names).
    """
    start1 = next(iter(graph.keys()))  # Choose an arbitrary start node from input graph.
    copy_char = "**"
    graph2 = {f"{key}{copy_char}": [f"{x}{copy_char}" for x in val] for key, val in graph.items()}
    for start2 in graph2:
        matched = FragNodeMatcher.match_nodes(graph, start1, graph2, start2)
        if start2 == f"{start1}{copy_char}":
            check.is_true(len(matched) == len(graph))
            [check.equal(f"{key}{copy_char}", val) for key, val in matched.items()]
        else:
            check.is_none(matched)


@pytest.mark.parametrize(
    ("smiles1", "smiles2", "n_expected"),
    [
        (["[U]CC[Np]", "[U]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN", "[Pu]CC(CC[Bk])[Np]", "[Bk]CO"], None, 2),
        (["[U]CC[Pu]", "[U]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN"], None, 2),
        (
            [
                "[U]CC[Np]",
                "[U]CC(C(C(F)(F)F)[Xe])[Pu]",
                "[Pu]CC(CC[Xe])[Np]",
            ],
            None,
            6,
        ),
        (
            ["[U]CC[Np]", "[U]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN", "[Pu]CC(CC[Bk])[Np]", "[Bk]CO"],
            ["[Xe]CC[Ar]", "[Xe]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN", "[Pu]CC(CC[Bk])[Ar]", "[Bk]CO"],
            2,
        ),
        (
            ["[U]CC[Pu]", "[U]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN"],
            [
                "[U]CC[Np]",
                "[U]CC(C(C(F)(F)F)[Xe])[Pu]",
                "[Pu]CC(CC[Xe])[Np]",
            ],
            0,
        ),
        # TODO: add more graph types for sanity checks. add different nodes1/2 for no matches.
    ],
)
@pytest.mark.parametrize("connectors", [{"U", "Np", "Pu", "Am", "Bk", "Xe", "Ar"}])
def test_match_all_graphs(smiles1, smiles2, connectors, n_expected):
    """Correct number of matches are found. Match dict contains the correct types."""
    if smiles2 is None:
        smiles2 = smiles1
    nodes1 = map_mol_frag_graph(smiles1, connectors)
    nodes2 = map_mol_frag_graph(smiles2, connectors)
    # Shift nodes2 order, should not affect the results.
    nodes2 = nodes2[2:] + nodes2[:2]
    nodes2.reverse()
    res = FragNodeMatcher.match_all_graphs(nodes1, nodes2)
    check.equal(len(res), n_expected)
    if res:
        for key, val in res[0].items():
            check.is_true(isinstance(key, FragNode))
            check.is_true(isinstance(val, FragNode))


@pytest.mark.parametrize(
    ("smiles", "cross_score", "expected"),
    [
        (
            ["[U]CC[Np]", "[U]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN", "[Pu]CC(CC[Bk])[Np]", "[Bk]CO"],
            True,
            [
                {"He": ["U"], "Ne": ["Np"], "Kr": ["Am"], "Ar": ["Pu"], "Xe": ["Bk"]},
                {"Ne": ["U"], "He": ["Np"], "Xe": ["Am"], "Ar": ["Pu"], "Kr": ["Bk"]},
            ],
        ),
        (
            ["[U]CC[Np]", "[U]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN", "[Pu]CC(CC[Bk])[Np]", "[Bk]CO"],
            False,
            [
                {"He": ["U"], "Ne": ["Np"], "Kr": ["Am"], "Ar": ["Pu"], "Xe": ["Bk"]},
                {"Ne": ["U"], "He": ["Np"], "Xe": ["Am"], "Ar": ["Pu"], "Kr": ["Bk"]},
            ],
        ),
        (
            ["[U]CC[Pu]", "[U]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN"],
            True,
            [
                {"Ar": ["U"], "He": ["Pu"], "Kr": ["Am"]},
                {"Ar": ["Pu"], "He": ["U"], "Kr": ["Am"]},
                {"Kr": ["U", "Pu"], "Ar": ["Am"], "He": ["Am"]},
            ],
        ),
        (
            ["[U]CC[Pu]", "[U]CC(C(C(F)(F)F)[Am])[Pu]", "[Am]CCN"],
            False,
            [
                {"Ar": ["U"], "He": ["Pu"], "Kr": ["Am"]},
                {"Ar": ["Pu"], "He": ["U"], "Kr": ["Am"]},
            ],
        ),
        (
            [
                "[U]CC[Np]",
                "[U]CC(C(C(F)(F)F)[Xe])[Pu]",
                "[Pu]CC(CC[Xe])[Np]",
            ],
            True,
            [
                {"He": ["U"], "Ne": ["Np"], "Xe": ["Pu"], "Ar": ["Pu"]},
                {"He": ["U"], "Xe": ["Np"], "Ar": ["Np"], "Ne": ["Pu"]},
                {"Ne": ["U"], "Xe": ["Np"], "Ar": ["Np"], "He": ["Pu"]},
                {"Xe": ["U"], "Ar": ["U"], "Ne": ["Np"], "He": ["Pu"]},
                {"Xe": ["U"], "Ar": ["U"], "He": ["Np"], "Ne": ["Pu"]},
                {"Ne": ["U"], "He": ["Np"], "Xe": ["Pu"], "Ar": ["Pu"]},
            ],
        ),
        # TODO: add more graph types for sanity checks.
        # ( # For scoring single known input query fragments.
        #   # As expected, match_all_graphs doesn't work with incomplete graphs, yet.
        #     ['CC[U]'],
        #     False,
        #     [{'He': ['U']}],
        # ),
        # (
        #     ['[Np]CCC[U]'],
        #     False,
        #     [
        #         {'Ne': ['Np'], 'He': ['U']},
        #         {'He': ['Np'], 'Ne': ['U']}
        #     ]
        # )
    ],
)
@pytest.mark.parametrize("connectors", [{"U", "Np", "Pu", "Am", "Bk"}])
def test_enumerate_fragment_connector_substitution(smiles, connectors, cross_score, expected):
    """Correct patterns of connector atom subsitution are found."""

    def replace_chars(string, subdict):
        for key, val in subdict.items():
            string = string.replace(key, val)
        return string

    def hash_subdict(results: list[dict]) -> list:
        def sort_values(d: dict) -> dict:
            return {k: sorted(v) for k, v in d.items()}

        def hash_dict(d: dict) -> str:
            return json.dumps(sort_values(d), sort_keys=True)

        return sorted([hash_dict(d) for d in results])

    subdict = {"U": "He", "Np": "Ne", "Pu": "Ar", "Am": "Kr", "Bk": "Xe"}
    frag_smiles = [replace_chars(smi, subdict) for smi in smiles]  # create fake fragments
    # print(frag_smiles)
    syn_nodes = map_mol_frag_graph(smiles, connectors)
    frag_nodes = map_mol_frag_graph(frag_smiles, connector_set=set(subdict.values()))
    matches = FragNodeMatcher.match_all_graphs(syn_nodes, frag_nodes)

    all_res = []
    for match in matches:
        res = enumerate_fragment_connector_substitution(syn_nodes, match, cross_score)
        for r in res:
            if r:
                all_res.append(r)
    check.equal(hash_subdict(all_res), hash_subdict(expected))
    # for r in all_res:
    #     print(r)


@pytest.mark.parametrize(
    ("sub_map", "expected_smi"),
    [
        ({"U": ["Ne"], "Np": ["Ar"]}, r"Cc1cc(on1)/C(=C/[5Ne])/[5Ar]"),
        ({"U": ["Ne"], "Np": ["Ne"]}, r"[H]/[C]=[C]\c1cc(no1)C.[5Ne]"),
        ({"U": ["He", "Ne"]}, r"[5He]/C=C(/c1cc(no1)C)\[Np].[5Ne]"),
    ],
)
def test_special_substitute_atoms(sub_map, expected_smi):
    """Atoms are substituted correctly, including placing centroids or overlapping connectors."""
    # Not testing the exact geometry of the centroids yet.
    mol = load_first_mol(DATA_DIR / "rocs_test_conf1.sdf")
    mol = special_substitute_atoms(mol, sub_map, isotope=5)
    check.equal(oechem.OEMolToSmiles(mol), expected_smi)


@pytest.mark.parametrize(
    ("synthon_smis", "smirks", "expected_res"),
    [
        (
            {
                0: "COCC(C)N=C(S[U])N([Np])N=[Pu]",
                1: "[O-][N+](=O)c1ccc(C=[Pu])o1",
                2: "Clc1ccccc1C([Np])=C[U]",
            },
            "([U]-[*:1].[Np]-[*:2]-[*:3]=[Pu]).[Pu]=[*:4].[Np]-[*:5]=[*:6]-[U]>>[*:1]-[*:6]=[*:5]-[*:2]-[*:3]=[*:4]",
            {"U": str(True), "Np": str(True), "Pu": str(False)},
        ),
        (
            {
                0: "CCOc1ccccc1N=C([Pu])S[U]",
                1: "[Np]N([Pu])C1CCCCC1",
                2: "Cc1cc(C([Np])=C[U])c(C)n1-c1ccc(C)c(C)c1",
            },
            "[U]-[*:1]-[*:2]-[Pu].[Np]-[*:3]-[Pu].[U]-[*:5]=[*:4]-[Np]>>[*:2]-1-[*:3]-[*:4]=[*:5]-[*:1]-1",
            {"U": str(True), "Np": str(True), "Pu": str(True)},
        ),
        (
            {
                0: "CC(C)N[U]",
                1: "CC([U])=O",
            },
            "[U]-[*:1].[U]-[*:2]>>[*:1]-[*:2]",
            {"U": str(False)},
        ),
        (
            {
                0: "CC1CN([U])CC2(CCN([Np])C2)O1",
                1: "CCC([U])=O",
                2: "CCC([Np])=O",
            },
            "([U]-[*:2].[Np]-[*:3]).[U]-[*:1].[Np]-[*:4]>>([*:1]-[*:2].[*:3]-[*:4])",
            {"U": str(False), "Np": str(False)},
        ),
    ],
)
@pytest.mark.parametrize("connectors", [["U", "Np", "Pu", "Am"]])
def test_label_synthon_ring_connector(synthon_smis, connectors, smirks, expected_res):
    """Connector atoms are labeled correctly as either ring or non-ring forming."""
    res = label_synthon_ring_connector(synthon_smis, connectors, smirks)
    check.equal(res, expected_res)


@pytest.mark.parametrize(
    (
        "smiles",
        "start_atom",
        "start_isotope",
        "end_atom",
        "end_isotope",
        "include_termini",
        "expected_res",
        "should_raise",
    ),
    [
        (
            "O=S(CC1=NC(C2=CC=C([U])N=C2)=C(N3C[C@H](C(NC4CCC4)=O)CC3[Np])O1)(NC(C)C)=O",
            "U",
            0,
            "Np",
            0,
            True,
            10,
            False,
        ),
        (
            "O=S(CC1=NC(C2=CC=C([U])N=C2)=C(N3C[C@H](C(NC4CCC4)=O)CC3[Np])O1)(NC(C)C)=O",
            "U",
            0,
            "Np",
            0,
            False,
            8,
            False,
        ),
        (
            "O=S(CC1=NC(C2=CC=C([333U])N=C2)=C(N3C[C@H](C(NC4CCC4)=O)CC3[128F])O1)(NC(C)C)=O",
            "U",
            333,
            "F",
            128,
            True,
            10,
            False,
        ),
        (
            "O=S(CC1=NC(C2=CC=C([333U])N=C2)=C(N3C[C@H](C(NC4CCC4)=O)CC3[128F])O1)(NC(C)C)=O",
            "U",
            333,
            "F",
            111,
            True,
            None,
            False,
        ),
        (  # Duplicate start atoms found.
            "O=S(CC1=NC(C2=CC=C([333U])N=C2)=C(N3C[C@H](C(NC4CCC4)=O)CC3[128F])O1)(NC(C)C)=O",
            "C",
            0,
            "F",
            111,
            True,
            None,
            True,
        ),
        (  # No start atom found.
            "O=S(CC1=NC(C2=CC=C([333U])N=C2)=C(N3C[C@H](C(NC4CCC4)=O)CC3[128F])O1)(NC(C)C)=O",
            "C",
            14,
            "F",
            111,
            True,
            None,
            True,
        ),
    ],
)
def test_get_shortest_path_atoms(
    smiles, start_atom, start_isotope, end_atom, end_isotope, include_termini, expected_res, should_raise
):
    """Corrected length of path is found."""
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    if should_raise:
        with check.raises(ValueError):
            get_shortest_path_atoms(mol, start_atom, start_isotope, end_atom, end_isotope, include_termini)
    else:
        atom_path = get_shortest_path_atoms(mol, start_atom, start_isotope, end_atom, end_isotope, include_termini)
        if atom_path is None:
            check.equal(None, expected_res)
        else:
            check.equal(len(atom_path), expected_res)


@pytest.mark.parametrize(
    ("smiles", "expected_res"),
    [
        (
            "OC(C[C@H](NC(OCC1C2=C[19C]=[20C]C=C2C3=CC=CC=C13)=O)CF)=O",
            6,
        ),
        (
            "OC(C[C@H](NC(OC[19C]1[20C]2=CC=CC=C2C3=CC=CC=C13)=O)CF)=O",
            5,
        ),
        (
            "[19C]12CCCCC[20C]1CCCCCC2",
            7,
        ),
        (
            "C1CCC[19C]2(CCCCCC3)[20C]3(C2)C1",
            3,
        ),
        (
            "[19C]1[20C]CCC2(CCCCCC3)C3(C2)C1",
            7,
        ),
        (
            "C12CC[19C][20C]CC(C2)CCCCCC1",
            8,
        ),
    ],
)
@pytest.mark.parametrize("include_self", [True, False])
def test_get_same_ring_bonds(
    smiles,
    include_self,
    expected_res,
):
    """Corrected number of ring bonds are found.

    Test examples have [19C] and [20C] as the start and end atoms of bond of interest.
    This function does not guard against null returns yet.
    """
    start_atom, start_isotope = "C", 19
    end_atom, end_isotope = "C", 20
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    oechem.OEFindRingAtomsAndBonds(mol)
    start = next(find_atoms(mol, start_atom, start_isotope))
    end = next(find_atoms(mol, end_atom, end_isotope))
    bond = mol.GetBond(start, end)
    ring_bonds = get_same_ring_bonds(bond, include_self)
    check.equal(len(ring_bonds), expected_res - int(not include_self))  # If not including self, subtract 1.


@pytest.mark.parametrize("distance", [1, 3, 7])
@pytest.mark.parametrize(("conn_atom_symbol", "conn_atom_isotope"), [("U", 0), ("Np", 0)])
@pytest.mark.parametrize(("new_atom_symbol", "new_atom_isotope"), [("Xe", 333), ("Se", 334)])
def test_extend_dummy_bond(
    mc_mol_2conn, distance, conn_atom_symbol, conn_atom_isotope, new_atom_symbol, new_atom_isotope
):
    """New atom is placed at the correct distance from the connector atom."""
    mol = mc_mol_2conn
    new_mol = extend_dummy_bond(
        mol,
        conn_atom_symbol,
        conn_atom_isotope,
        new_atom_symbol,
        new_atom_isotope,
        distance,
    )

    check.equal(mol.NumConfs(), new_mol.NumConfs())
    for conf in new_mol.GetConfs():
        original_conn = next(find_atoms(conf, conn_atom_symbol, conn_atom_isotope))
        new_atom = next(find_atoms(conf, new_atom_symbol, new_atom_isotope))
        bond_dist = oechem.OEGetDistance(conf, original_conn, new_atom)
        check.almost_equal(bond_dist, distance, abs=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main())
