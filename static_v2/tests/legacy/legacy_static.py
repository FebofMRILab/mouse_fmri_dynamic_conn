"""Verbatim copies of the previous custom functions from
full_analysis_abeta_mice_permutation_transitivity.py (threshold, nodal
efficiency, NBS), for the equivalence tests only. set_size(n) defines the
module globals the functions use."""
import numpy as np

N_ROI = None
N_EDGES = None
triu_i = triu_j = None


def set_size(n):
    global N_ROI, N_EDGES, triu_i, triu_j
    N_ROI = n
    triu_i, triu_j = np.triu_indices(n, k=1)
    N_EDGES = len(triu_i)


def proportional_threshold_positive(matrix, density):
    """
    Retain the strongest positive edges so that the requested
    density refers to all possible undirected edges.

    Returns
    -------
    weighted : symmetric weighted adjacency matrix
    binary : symmetric binary adjacency matrix
    """
    matrix = np.asarray(matrix, dtype=float).copy()
    matrix = np.nan_to_num(matrix, nan=0.0)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)

    edge_weights = matrix[triu_i, triu_j]
    positive_indices = np.where(edge_weights > 0)[0]

    n_keep = int(np.ceil(density * N_EDGES))
    n_keep = min(n_keep, len(positive_indices))

    weighted = np.zeros((N_ROI, N_ROI), dtype=float)

    if n_keep == 0:
        return weighted, weighted.copy()

    ranked_positive = positive_indices[
        np.argsort(edge_weights[positive_indices])[::-1]
    ]
    selected = ranked_positive[:n_keep]

    selected_i = triu_i[selected]
    selected_j = triu_j[selected]
    selected_weights = edge_weights[selected]

    weighted[selected_i, selected_j] = selected_weights
    weighted[selected_j, selected_i] = selected_weights

    binary = (weighted > 0).astype(float)

    return weighted, binary


from scipy.sparse.csgraph import shortest_path


def nodal_efficiency(binary):
    """
    Nodal efficiency (Achard & Bullmore, 2007): mean inverse shortest-path
    length from each node to all other nodes on the binary graph.
    Averaging over nodes gives networkx.global_efficiency, so this matches
    the global efficiency in Figure 6. Isolated nodes get 0.
    """
    distances = shortest_path(binary, method="D", unweighted=True, directed=False)
    with np.errstate(divide="ignore"):
        inverse = 1.0 / distances
    np.fill_diagonal(inverse, 0.0)
    inverse[~np.isfinite(inverse)] = 0.0
    return inverse.sum(axis=1) / (binary.shape[0] - 1)


from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def welch_t_vectorized(group_a_edges, group_b_edges):
    """
    Edgewise Welch t statistics (subjects x edges inputs).
    Positive values indicate greater connectivity in group A.
    """
    group_a_edges = np.asarray(group_a_edges, dtype=float)
    group_b_edges = np.asarray(group_b_edges, dtype=float)

    n_a = group_a_edges.shape[0]
    n_b = group_b_edges.shape[0]

    mean_a = np.nanmean(group_a_edges, axis=0)
    mean_b = np.nanmean(group_b_edges, axis=0)

    variance_a = np.nanvar(group_a_edges, axis=0, ddof=1)
    variance_b = np.nanvar(group_b_edges, axis=0, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = (mean_a - mean_b) / np.sqrt(
            variance_a / n_a + variance_b / n_b
        )

    return np.nan_to_num(t_values, nan=0.0, posinf=0.0, neginf=0.0)


def _label_suprathreshold_edges(t_values, threshold, tail):
    """
    One-sided suprathreshold set.
    tail = +1 keeps t > threshold  (A > B contrast)
    tail = -1 keeps t < -threshold (A < B contrast)

    Returns the suprathreshold edge indices and the connected-component
    label of each of those edges.
    """
    suprathreshold = (tail * np.asarray(t_values)) > threshold
    supra_idx = np.flatnonzero(suprathreshold)

    if supra_idx.size == 0:
        return supra_idx, None

    adjacency = coo_matrix(
        (
            np.ones(supra_idx.size),
            (triu_i[supra_idx], triu_j[supra_idx]),
        ),
        shape=(N_ROI, N_ROI),
    )

    _, node_labels = connected_components(adjacency, directed=False)

    # Both endpoints of an edge share a label.
    return supra_idx, node_labels[triu_i[supra_idx]]


def max_component_extent(t_values, threshold, tail):
    """Largest component extent (number of edges); used in permutations."""
    _, edge_labels = _label_suprathreshold_edges(t_values, threshold, tail)

    if edge_labels is None:
        return 0

    return int(np.bincount(edge_labels).max())


def suprathreshold_components(t_values, threshold, tail):
    """
    Connected components of a one-sided suprathreshold network.
    Component extent is the number of suprathreshold edges.
    """
    supra_idx, edge_labels = _label_suprathreshold_edges(
        t_values, threshold, tail
    )

    if edge_labels is None:
        return []

    components = []

    for label in np.unique(edge_labels):
        edge_indices = supra_idx[edge_labels == label]
        nodes = np.unique(
            np.concatenate([triu_i[edge_indices], triu_j[edge_indices]])
        )

        components.append({
            "nodes": nodes.tolist(),
            "edge_indices": edge_indices,
            "extent": int(edge_indices.size),
            "tail": "positive" if tail > 0 else "negative",
        })

    components.sort(key=lambda c: c["extent"], reverse=True)

    return components


NBS_TAILS = {"positive": 1, "negative": -1}


def run_nbs(
    matrices_group_a,
    matrices_group_b,
    primary_threshold=3.1,
    n_permutations=5000,
    random_seed=20260911,
):
    """
    Standard NBS: two one-sided contrasts (A > B and A < B), each with
    its own permutation null of the maximum component extent.

    Returns
    -------
    observed_t : array
        Edgewise Welch t statistics (A - B).
    components : list of dicts
        Observed components from both contrasts, each with
        'tail' and a within-contrast FWER p-value.
    null_max_extents : dict
        {'positive': array, 'negative': array}
    """
    edge_a = matrices_group_a[:, triu_i, triu_j]
    edge_b = matrices_group_b[:, triu_i, triu_j]

    observed_t = welch_t_vectorized(edge_a, edge_b)

    observed_components = {
        name: suprathreshold_components(observed_t, primary_threshold, tail)
        for name, tail in NBS_TAILS.items()
    }

    combined = np.vstack([edge_a, edge_b])
    n_group_a = edge_a.shape[0]
    n_total = combined.shape[0]

    rng = np.random.default_rng(random_seed)
    null_max_extents = {
        name: np.zeros(n_permutations, dtype=int) for name in NBS_TAILS
    }

    for permutation in range(n_permutations):
        shuffled = rng.permutation(n_total)

        permuted_t = welch_t_vectorized(
            combined[shuffled[:n_group_a]],
            combined[shuffled[n_group_a:]],
        )

        # Same relabelling feeds both contrasts.
        for name, tail in NBS_TAILS.items():
            null_max_extents[name][permutation] = max_component_extent(
                permuted_t, primary_threshold, tail
            )

        if (permutation + 1) % 1000 == 0 or permutation == n_permutations - 1:
            print(f"    Permutation {permutation + 1} of {n_permutations}")

    components = []

    for name in NBS_TAILS:
        null = null_max_extents[name]

        for component in observed_components[name]:
            component["p_FWER"] = (
                1 + np.sum(null >= component["extent"])
            ) / (n_permutations + 1)
            components.append(component)

    components.sort(key=lambda c: c["extent"], reverse=True)

    return observed_t, components, null_max_extents
