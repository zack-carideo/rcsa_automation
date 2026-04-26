from __future__ import annotations
import json
import pandas as pd


def build_prc_graph(
    df: pd.DataFrame,
    *,
    process_col: str = "process",
    risk_col: str = "risk",
    control_col: str = "control",
) -> dict[str, list[dict]]:
    """
    Convert a flat Process–Risk–Control DataFrame into a network graph dict.

    Handles M:M:M relationships automatically — shared nodes are deduplicated,
    duplicate edges across rows produce a single link.

    Parameters
    ----------
    df : pd.DataFrame
        Each row represents one (process, risk, control) triplet.
    process_col, risk_col, control_col : str
        Column name overrides (default: 'process', 'risk', 'control').

    Returns
    -------
    dict with:
        nodes : list[dict] → {id: str, label: str, type: str}
        links : list[dict] → {source: str, target: str}
    """
    col_map = {"process": process_col, "risk": risk_col, "control": control_col}

    # Normalize: select → rename → drop nulls → strip whitespace
    work = (
        df[list(col_map.values())]
        .rename(columns={v: k for k, v in col_map.items()})
        .dropna(how="any")
        .astype(str)
        .apply(lambda s: s.str.strip())
        .reset_index(drop=True)
    )

    PREFIX = {"process": "p", "risk": "r", "control": "c"}

    def _build_id_map(series: pd.Series, prefix: str) -> dict[str, str]:
        """Preserve first-seen order; assign stable positional IDs."""
        seen: dict[str, str] = {}
        for label in series:
            if label not in seen:
                seen[label] = f"{prefix}{len(seen)}"
        return seen

    id_map = {
        entity: _build_id_map(work[entity], PREFIX[entity])
        for entity in ("process", "risk", "control")
    }

    nodes = [
        {"id": node_id, "label": label, "type": entity}
        for entity, mapping in id_map.items()
        for label, node_id in mapping.items()
    ]

    def _make_links(src_type: str, tgt_type: str) -> list[dict]:
        return [
            {
                "source": id_map[src_type][getattr(row, src_type)],
                "target": id_map[tgt_type][getattr(row, tgt_type)],
            }
            for row in (
                work[[src_type, tgt_type]]
                .drop_duplicates()
                .itertuples(index=False)
            )
        ]

    links = _make_links("process", "risk") + _make_links("risk", "control")

    return {"nodes": nodes, "links": links}


def prc_graph_to_json(graph: dict, *, indent: int = 2) -> str:
    """Serialize graph dict to JSON string."""
    return json.dumps(graph, indent=indent)


def prc_graph_stats(
    df: pd.DataFrame,
    *,
    process_col: str = "process",
    risk_col: str = "risk",
    control_col: str = "control",
) -> pd.DataFrame:
    """
    Degree stats per entity — surface high-centrality controls and
    uncovered risks before visualizing.

    Returns a DataFrame sorted by type → degree descending.
    """
    graph = build_prc_graph(
        df, process_col=process_col, risk_col=risk_col, control_col=control_col
    )

    degree: dict[str, int] = {n["id"]: 0 for n in graph["nodes"]}
    for link in graph["links"]:
        degree[link["source"]] += 1
        degree[link["target"]] += 1

    return (
        pd.DataFrame([
            {"id": n["id"], "label": n["label"], "type": n["type"], "degree": degree[n["id"]]}
            for n in graph["nodes"]
        ])
        .sort_values(["type", "degree"], ascending=[True, False])
        .reset_index(drop=True)
    )
    

if __name__ == "__main__":
    
    # --- Any flat RCSA DataFrame shape ---
    df = pd.DataFrame({
        "process":  ["Loan Origination", "Loan Origination", "AML Screening", "Wire Transfer"],
        "risk":     ["Credit Risk",      "Fraud Risk",       "Fraud Risk",    "Sanctions Risk"],
        "control":  ["Credit Bureau",    "KYC Verification", "KYC Verification", "OFAC Screening"],
    })

    # One call → graph dict
    graph = build_prc_graph(df)

    # If your columns are named differently
    graph = build_prc_graph(df, process_col="Process_Name", risk_col="Risk_Desc", control_col="Control_ID")

    # Serialize for the widget — paste into rawNodes / rawLinks JS vars
    print(prc_graph_to_json(graph))

    # EDA before you visualize — spot orphaned controls, overloaded risks
    prc_graph_stats(df)
    #    id  label               type     degree
    # 0  p0  Loan Origination    process  2
    # 1  r0  Credit Risk         risk     2
    # 2  r1  Fraud Risk          risk     3      ← shared across 2 processes, 1 control
    # 3  c0  KYC Verification    control  2      ← mitigates Fraud Risk for 2 processes