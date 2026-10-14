import os
import sqlite3
from collections import defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
import networkx as nx

from mri.config import ROOT as BASE_DIR  # repository root
DB_PATH = os.path.join(BASE_DIR, "data", "marketplace.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "graph_risk_scores.csv")

# ---------------------------------------------------------------------------
# Shared-attribute linking policy
# ---------------------------------------------------------------------------
# A shared device or region is evidence only when it is RARE. The previous
# code capped region groups at a fixed 25 members, which on any realistic
# dataset excluded every group and emitted a permanently-zero feature that
# still carried weight in the score. Cardinality is therefore judged
# relatively, not against a hardcoded constant:
#
#   a group is informative  <=>  2 <= members <= max(MIN_INFORMATIVE_MEMBERS,
#                                                    MAX_GROUP_SHARE * population)
#
# This also bounds the O(k^2) clique expansion: group size scales with the
# population, so the edge count stays roughly linear in it instead of
# quadratic. MAX_GROUP_MEMBERS is a hard safety ceiling for very large
# populations; when it bites, it is logged, never silently applied.
MAX_GROUP_SHARE = 0.01
MIN_INFORMATIVE_MEMBERS = 10
MAX_GROUP_MEMBERS = 200


def log(msg: str):
    print(f"[graph_intelligence] {msg}")


def _check_query(query: str) -> str:
    """Accept only a single literal SELECT. Queries in this module are constants;
    this guard keeps that true if a caller is ever changed."""
    q = query.strip().rstrip(";")
    if ";" in q or not q.upper().startswith("SELECT"):
        raise ValueError("only a single SELECT statement is allowed")
    return q


def load_sql(query: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(_check_query(query), conn)


def normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def informative_groups(groups, population: int, label: str):
    """Yield (key, members) for shared-attribute groups that carry information.

    Returns the kept groups plus a diagnostics dict. A group is dropped when it
    has a single member (no link), when it covers too large a share of the
    population to individuate anybody, or when it exceeds the hard expansion
    ceiling. Every drop is counted so a dead signal can never pass as a zero.
    """
    max_members = max(MIN_INFORMATIVE_MEMBERS, int(MAX_GROUP_SHARE * population))
    max_members = min(max_members, MAX_GROUP_MEMBERS)

    kept, diagnostics = [], {
        "label": label, "total": 0, "kept": 0, "singleton": 0,
        "too_common": 0, "max_members": max_members, "largest": 0, "pairs": 0,
    }

    for key, members in groups.items():
        members = sorted(set(members))
        diagnostics["total"] += 1
        diagnostics["largest"] = max(diagnostics["largest"], len(members))
        if len(members) < 2:
            diagnostics["singleton"] += 1
            continue
        if len(members) > max_members:
            diagnostics["too_common"] += 1
            continue
        kept.append((key, members))
        diagnostics["kept"] += 1
        diagnostics["pairs"] += len(members) * (len(members) - 1) // 2

    return kept, diagnostics


def weighted_risk_score(df: pd.DataFrame, spec, log_fn):
    """Weighted sum over features, excluding any that carry no information.

    A constant (zero-variance) or all-NaN feature cannot rank anything: left in
    the sum it contributes a fixed 0 to every entity while still consuming its
    share of the weight, silently shrinking the score's dynamic range. Such
    features are dropped, logged, and the remaining weights renormalised to 1.
    """
    usable, dropped = [], []
    for column, weight in spec:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() == 0:
            dropped.append((column, weight, "not computable (all NaN)"))
        elif values.nunique(dropna=True) <= 1:
            dropped.append((column, weight, f"zero variance (constant {values.dropna().iloc[0]})"))
        else:
            usable.append((column, weight))

    for column, weight, reason in dropped:
        log_fn(f"  EXCLUDED from graph risk score: {column} (weight {weight:.2f}) - {reason}")

    if not usable:
        raise ValueError("no informative graph features; cannot compute a graph risk score")

    total_weight = sum(weight for _, weight in usable)
    if dropped:
        log_fn(f"  renormalising {len(usable)} remaining features over weight {total_weight:.2f}")

    score = pd.Series(0.0, index=df.index)
    for column, weight in usable:
        score = score + (weight / total_weight) * normalize(
            pd.to_numeric(df[column], errors="coerce")
        ).fillna(0.0)
    return score, [c for c, _ in usable], dropped


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log("Loading source tables...")
    orders = load_sql("SELECT order_id, customer_id, seller_id FROM orders")
    login = load_sql("SELECT user_id, device_id, ip_region FROM login_activity")

    if orders.empty and login.empty:
        raise ValueError("Orders and login_activity tables are empty. Cannot build graph intelligence.")

    G = nx.Graph()

    # Edge attributes are accumulated independently per pair before the graph
    # is built. The previous code tagged each edge with a single ``edge_type``
    # decided by whichever relationship happened to be seen first, so a
    # shared device between a customer and a seller they also transact with --
    # precisely the collusion pattern of interest -- was recorded as a plain
    # transaction and never counted as a device link.
    pair_attributes = defaultdict(
        lambda: {"transactions": 0.0, "shared_devices": 0, "shared_regions": 0}
    )
    node_types = {}

    def register(node: str, entity_type: str):
        node_types[node] = entity_type

    # -------------------------
    # Add customer-seller interactions
    # -------------------------
    log("Adding customer-seller interaction edges...")
    if not orders.empty:
        orders["customer_node"] = "customer_" + orders["customer_id"].astype(str)
        orders["seller_node"] = "seller_" + orders["seller_id"].astype(str)

        for node in orders["customer_node"].unique():
            register(node, "customer")
        for node in orders["seller_node"].unique():
            register(node, "seller")

        edge_counts = (
            orders.groupby(["customer_node", "seller_node"])
            .size()
            .reset_index(name="interaction_count")
        )
        for customer_node, seller_node, count in edge_counts.itertuples(index=False):
            key = (customer_node, seller_node) if customer_node < seller_node else (seller_node, customer_node)
            pair_attributes[key]["transactions"] += float(count)

    # -------------------------
    # Add shared-device and shared-region relationships
    # -------------------------
    log("Adding shared-attribute relationship edges...")
    group_diagnostics = {}
    if not login.empty:
        login["user_id"] = login["user_id"].astype(str)
        login["device_id"] = login["device_id"].astype(str)
        login["ip_region"] = login["ip_region"].astype(str)

        def map_user_node(uid: str) -> str:
            if uid.startswith("C"):
                return f"customer_{uid[1:]}"
            elif uid.startswith("S"):
                return f"seller_{uid[1:]}"
            return f"unknown_{uid}"

        login["entity_node"] = login["user_id"].apply(map_user_node)
        for node in login["entity_node"].unique():
            register(node, node.split("_", 1)[0])

        # Population = every entity the graph knows about, so the
        # informativeness threshold scales with the dataset.
        population = len(set(node_types))

        for attribute, column, attr_key in [
            ("device", "device_id", "shared_devices"),
            ("region", "ip_region", "shared_regions"),
        ]:
            groups = login.groupby(column)["entity_node"].unique()
            kept, diagnostics = informative_groups(groups, population, attribute)
            group_diagnostics[attribute] = diagnostics
            log(
                f"  {attribute}: {diagnostics['kept']}/{diagnostics['total']} groups informative "
                f"(<= {diagnostics['max_members']} members of {population}); "
                f"{diagnostics['too_common']} too common, {diagnostics['singleton']} singleton; "
                f"largest group {diagnostics['largest']}; {diagnostics['pairs']} candidate pairs"
            )
            for _, members in kept:
                for u, v in combinations(members, 2):
                    key = (u, v) if u < v else (v, u)
                    pair_attributes[key][attr_key] += 1

    for node, entity_type in node_types.items():
        G.add_node(node, entity_type=entity_type)

    for (u, v), attributes in pair_attributes.items():
        # edge_type is a frozenset of every relationship present on the edge;
        # the per-attribute counters are what the features actually read, so no
        # relationship can mask another.
        edge_types = frozenset(
            name for name, present in (
                ("transaction", attributes["transactions"] > 0),
                ("shared_device", attributes["shared_devices"] > 0),
                ("shared_region", attributes["shared_regions"] > 0),
            ) if present
        )
        G.add_edge(
            u, v,
            edge_type=edge_types,
            transactions=attributes["transactions"],
            shared_devices=attributes["shared_devices"],
            shared_regions=attributes["shared_regions"],
            # Relationship-weighted strength: transactions and shared devices
            # count fully, a shared (rare) region counts half, as before.
            weight=(
                attributes["transactions"]
                + attributes["shared_devices"]
                + 0.5 * attributes["shared_regions"]
            ),
        )

    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes. Check source data.")

    log(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    # -------------------------
    # Compute graph features
    # -------------------------
    log("Computing graph metrics...")
    degree_dict = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)
    connected_components = list(nx.connected_components(G))
    node_count = G.number_of_nodes()

    # Raw component size is useless whenever the data forms one giant
    # component (it is then the same constant for everyone). Local cohesion
    # measures still separate a tight ring from a hub, so they are computed
    # alongside it and the score uses the informative one.
    core_numbers = nx.core_number(G)
    clustering = nx.clustering(G)

    component_map = {}
    component_size_map = {}

    for idx, component in enumerate(connected_components):
        size = len(component)
        for node in component:
            component_map[node] = idx
            component_size_map[node] = size

    rows = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))

        shared_device_links = 0
        shared_region_links = 0
        transaction_links = 0
        edge_weight_sum = 0.0

        for neighbor in neighbors:
            edge_data = G.get_edge_data(node, neighbor, default={})
            edge_weight_sum += float(edge_data.get("weight", 0))

            # Each relationship on the edge is counted on its own; one edge can
            # legitimately be a transaction AND a shared-device link.
            if edge_data.get("shared_devices", 0) > 0:
                shared_device_links += 1
            if edge_data.get("shared_regions", 0) > 0:
                shared_region_links += 1
            if edge_data.get("transactions", 0) > 0:
                transaction_links += 1

        if node.startswith("customer_"):
            entity_type = "customer"
            entity_id = node.replace("customer_", "")
        elif node.startswith("seller_"):
            entity_type = "seller"
            entity_id = node.replace("seller_", "")
        else:
            entity_type = "unknown"
            entity_id = node

        rows.append({
            "entity_id": entity_id,
            "entity_type": entity_type,
            "graph_degree": degree_dict.get(node, 0),
            "degree_centrality": degree_centrality.get(node, 0),
            "component_id": component_map.get(node, -1),
            "component_size": component_size_map.get(node, 1),
            "component_size_ratio": component_size_map.get(node, 1) / node_count,
            "core_number": core_numbers.get(node, 0),
            "clustering_coefficient": clustering.get(node, 0.0),
            "shared_device_links": shared_device_links,
            "shared_region_links": shared_region_links,
            "transaction_links": transaction_links,
            "edge_weight_sum": edge_weight_sum
        })

    graph_df = pd.DataFrame(rows)

    # A shared-attribute feature whose groups were ALL uninformative was never
    # measured; recording it as 0 would be indistinguishable from "measured and
    # found nothing". NaN keeps the difference visible, and the scorer below
    # excludes it from the weighted sum instead of multiplying a dead signal.
    for attribute, column in [("device", "shared_device_links"),
                              ("region", "shared_region_links")]:
        diagnostics = group_diagnostics.get(attribute)
        if diagnostics is not None and diagnostics["kept"] == 0:
            log(f"  NOTE: no informative {attribute} group in this dataset "
                f"(largest has {diagnostics['largest']} members, threshold "
                f"{diagnostics['max_members']}); {column} is not measurable, recorded as NaN")
            graph_df[column] = np.nan

    # -------------------------
    # Graph risk score fusion
    # -------------------------
    log("Computing graph risk score...")
    graph_df["norm_degree"] = normalize(graph_df["graph_degree"])
    graph_df["norm_component_size"] = normalize(graph_df["component_size"])
    graph_df["norm_core_number"] = normalize(graph_df["core_number"])
    graph_df["norm_clustering"] = normalize(graph_df["clustering_coefficient"])
    graph_df["norm_shared_device_links"] = normalize(
        pd.to_numeric(graph_df["shared_device_links"], errors="coerce"))
    graph_df["norm_shared_region_links"] = normalize(
        pd.to_numeric(graph_df["shared_region_links"], errors="coerce"))
    graph_df["norm_edge_weight_sum"] = normalize(graph_df["edge_weight_sum"])

    # Most of the old 0.25 "component size" weight moves to k-core depth,
    # which separates entities inside a dense mutually-connected ring from
    # high-degree hubs and stays informative when the graph is one component.
    # Component size is kept in the sum as a RATIO (comparable across dataset
    # sizes) and is dropped automatically, with a log line, on the datasets
    # where it is the same constant for every entity.
    score_spec = [
        ("graph_degree", 0.30),
        ("core_number", 0.15),
        ("component_size_ratio", 0.10),
        ("shared_device_links", 0.25),
        ("shared_region_links", 0.10),
        ("edge_weight_sum", 0.10),
    ]
    graph_df["graph_risk_score"], used_features, dropped_features = weighted_risk_score(
        graph_df, score_spec, log
    )
    log(f"  graph risk score computed from {used_features}")

    graph_df["graph_risk_band"] = pd.cut(
        graph_df["graph_risk_score"],
        bins=[-0.01, 0.30, 0.60, 1.00],
        labels=["Low", "Medium", "High"]
    )

    graph_df = graph_df.sort_values("graph_risk_score", ascending=False).reset_index(drop=True)
    graph_df.to_csv(OUTPUT_PATH, index=False)

    log(f"Saved graph risk scores to: {OUTPUT_PATH}")
    print(graph_df.head(10))


if __name__ == "__main__":
    main()