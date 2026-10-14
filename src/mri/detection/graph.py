import os
import sqlite3
import pandas as pd
import numpy as np
import networkx as nx

from mri.config import ROOT as BASE_DIR  # repository root
DB_PATH = os.path.join(BASE_DIR, "data", "marketplace.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "graph_risk_scores.csv")


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

    # -------------------------
    # Add customer-seller interactions
    # -------------------------
    log("Adding customer-seller interaction edges...")
    if not orders.empty:
        orders["customer_node"] = "customer_" + orders["customer_id"].astype(str)
        orders["seller_node"] = "seller_" + orders["seller_id"].astype(str)

        edge_counts = (
            orders.groupby(["customer_node", "seller_node"])
            .size()
            .reset_index(name="interaction_count")
        )

        for _, row in edge_counts.iterrows():
            G.add_node(row["customer_node"], entity_type="customer")
            G.add_node(row["seller_node"], entity_type="seller")
            G.add_edge(
                row["customer_node"],
                row["seller_node"],
                edge_type="transaction",
                weight=float(row["interaction_count"])
            )

    # -------------------------
    # Add shared-device relationships
    # -------------------------
    log("Adding shared-device relationship edges...")
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

        # Shared device edges
        device_groups = login.groupby("device_id")["entity_node"].unique()
        for device_id, members in device_groups.items():
            members = list(set(members))
            if len(members) > 1:
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        u, v = members[i], members[j]
                        if G.has_edge(u, v):
                            if "shared_devices" not in G[u][v]:
                                G[u][v]["shared_devices"] = 0
                            G[u][v]["shared_devices"] += 1
                        else:
                            G.add_edge(
                                u, v,
                                edge_type="shared_device",
                                weight=1.0,
                                shared_devices=1
                            )

        # Shared region edges
        region_groups = login.groupby("ip_region")["entity_node"].unique()
        for region, members in region_groups.items():
            members = list(set(members))
            if len(members) > 1 and len(members) <= 25:  # keep graph practical
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        u, v = members[i], members[j]
                        if G.has_edge(u, v):
                            if "shared_regions" not in G[u][v]:
                                G[u][v]["shared_regions"] = 0
                            G[u][v]["shared_regions"] += 1
                        else:
                            G.add_edge(
                                u, v,
                                edge_type="shared_region",
                                weight=0.5,
                                shared_regions=1
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

            if edge_data.get("edge_type") == "shared_device":
                shared_device_links += 1
            elif edge_data.get("edge_type") == "shared_region":
                shared_region_links += 1
            elif edge_data.get("edge_type") == "transaction":
                transaction_links += 1
            else:
                if edge_data.get("shared_devices", 0) > 0:
                    shared_device_links += 1
                if edge_data.get("shared_regions", 0) > 0:
                    shared_region_links += 1

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
            "shared_device_links": shared_device_links,
            "shared_region_links": shared_region_links,
            "transaction_links": transaction_links,
            "edge_weight_sum": edge_weight_sum
        })

    graph_df = pd.DataFrame(rows)

    # -------------------------
    # Graph risk score fusion
    # -------------------------
    log("Computing graph risk score...")
    graph_df["norm_degree"] = normalize(graph_df["graph_degree"])
    graph_df["norm_component_size"] = normalize(graph_df["component_size"])
    graph_df["norm_shared_device_links"] = normalize(graph_df["shared_device_links"])
    graph_df["norm_shared_region_links"] = normalize(graph_df["shared_region_links"])
    graph_df["norm_edge_weight_sum"] = normalize(graph_df["edge_weight_sum"])

    graph_df["graph_risk_score"] = (
        0.30 * graph_df["norm_degree"] +
        0.25 * graph_df["norm_component_size"] +
        0.25 * graph_df["norm_shared_device_links"] +
        0.10 * graph_df["norm_shared_region_links"] +
        0.10 * graph_df["norm_edge_weight_sum"]
    )

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