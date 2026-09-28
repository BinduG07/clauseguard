"""
STEP 6: Full Pipeline (LangGraph)
-----------------------------------
Wires Extractor -> Retriever -> Risk-Flagger -> Summarizer into a single
LangGraph graph, mirroring the CodeCrew Planner/Coder/Reviewer/Tester
structure.
"""

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

from extractor import extract_clauses
from embeddings import ClauseRetriever
from risk_flagger import RiskFlagger


class ReviewState(TypedDict):
    contract_text: str
    clauses: List[Dict]
    retrieved: List[Dict]
    flagged: List[Dict]
    report: str


retriever = ClauseRetriever()
flagger = RiskFlagger()


def extractor_node(state: ReviewState) -> ReviewState:
    clauses = extract_clauses(state["contract_text"])
    return {**state, "clauses": clauses}


def retriever_node(state: ReviewState) -> ReviewState:
    retrieved = []
    for clause in state["clauses"]:
        matches = retriever.query(clause["clause_text"], k=1)
        retrieved.append({**clause, "match": matches[0] if matches else None})
    return {**state, "retrieved": retrieved}


def risk_flagger_node(state: ReviewState) -> ReviewState:
    flagged = []
    for item in state["retrieved"]:
        result = flagger.flag(item["clause_text"])
        flagged.append({**item, "risk": result})
    return {**state, "flagged": flagged}


def summarizer_node(state: ReviewState) -> ReviewState:
    lines = ["# ClauseGuard Risk Report\n"]
    risk_order = {"High-Risk": 0, "Needs-Human-Review": 1, "Unusual": 2, "Missing": 3, "Standard": 4}
    sorted_items = sorted(
        state["flagged"], key=lambda x: risk_order.get(x["risk"]["label"], 5)
    )
    for item in sorted_items:
        lines.append(f"## {item['heading']}")
        lines.append(f"**Risk:** {item['risk']['label']}  (source: {item['risk']['source']})")
        if item.get("match"):
            lines.append(f"*Closest standard clause type:* {item['match']['matched_clause_type']} "
                         f"(similarity {item['match']['similarity']:.2f})")
        lines.append(f"> {item['clause_text'][:300]}...\n")
    report = "\n".join(lines)
    return {**state, "report": report}


def build_graph():
    graph = StateGraph(ReviewState)
    graph.add_node("extractor", extractor_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("risk_flagger", risk_flagger_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("extractor")
    graph.add_edge("extractor", "retriever")
    graph.add_edge("retriever", "risk_flagger")
    graph.add_edge("risk_flagger", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    sample_contract = """
    1. Confidentiality
    Each party agrees to keep all shared information confidential for 5 years.

    2. Liability
    The Company shall have unlimited liability for any breach of this agreement.

    3. Governing Law
    This agreement shall be governed by the laws of the State of Delaware.
    """
    result = app.invoke({"contract_text": sample_contract})
    print(result["report"])
