"""LangGraph wiring for the ReAct agentic loop.

Graph structure
───────────────
  agent  ──[should_continue]──► tool_executor ──► agent  (loop)
    │
    └──[should_continue = "end"]──► END

The agent node calls Claude with tool schemas. Claude decides what to do next.
tool_executor runs the chosen tool and appends the result to the message history.
The loop repeats until Claude calls a terminal tool (answer_directly or
request_clarification) or the iteration cap is hit.

Old fixed pipeline (removed):
  planner → retriever → synthesizer → critic → [retry or end]
"""
from functools import partial

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from src.agents.llm import BedrockLLM
from src.agents.nodes import agent_node, should_continue, tool_executor_node
from src.agents.state import AgentState
from src.agents.tools import ToolExecutor
from src.retrieval.embeddings import BedrockEmbeddings
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import Reranker


def build_agent(db: Session):
    """Build a compiled LangGraph ReAct agent with per-request dependencies."""
    llm = BedrockLLM()
    embedder = BedrockEmbeddings()
    retriever = HybridRetriever(db, embedder)
    reranker = Reranker()
    executor = ToolExecutor(retriever, reranker)

    graph = StateGraph(AgentState)

    graph.add_node("agent", partial(agent_node, llm=llm))
    graph.add_node("tool_executor", partial(tool_executor_node, executor=executor))

    graph.set_entry_point("agent")

    # After each agent turn: either run the chosen tool or finish
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "execute_tool": "tool_executor",
            "end": END,
        },
    )

    # After executing a tool: always return to the agent for the next decision
    graph.add_edge("tool_executor", "agent")

    return graph.compile()
