"""LangGraph pipeline: node definitions and graph assembly.

The pipeline is a linear three-node graph:

    load ──► summarize ──► save

Each node receives the full :class:`~call_summarizer.models.SummaryState` dict,
updates the fields it owns, and passes it forward. An ``error`` field acts as a
short-circuit: downstream nodes skip their work when it is set.
"""

import logging
from pathlib import Path
from typing import Callable

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .models import SummaryState
from .storage import save_summary
from .summarizer import CHAR_LIMIT, generate_summary, validate_summary_length
from .transcript import load_transcript

logger = logging.getLogger(__name__)


def _load_node(state: SummaryState) -> SummaryState:
    """Pipeline node: load transcript text from disk into state.

    Args:
        state: Current pipeline state containing ``transcript_path``.

    Returns:
        Updated state with ``transcript_content`` populated, or ``error`` set
        if the file could not be loaded.
    """
    path = Path(state["transcript_path"])
    logger.debug("Load node — reading: %s", path)
    try:
        content = load_transcript(path)
        return {**state, "transcript_content": content}
    except Exception as exc:
        logger.error("Load node failed for %s: %s", path.name, exc)
        return {**state, "error": str(exc), "transcript_content": ""}


def _make_summarize_node(llm: ChatGroq) -> Callable[[SummaryState], SummaryState]:
    """Return a summarize node bound to *llm*.

    Using a factory keeps the LLM instance out of the module-level scope,
    making each call to :func:`build_graph` produce an independent node with
    its own LLM reference.

    Args:
        llm: A configured Groq LLM client to use for inference.

    Returns:
        A callable that acts as the summarize node in the LangGraph pipeline.
    """

    def _summarize_node(state: SummaryState) -> SummaryState:
        """Pipeline node: generate a summary from the transcript via the LLM.

        Args:
            state: Current pipeline state containing ``transcript_content``.

        Returns:
            Updated state with ``summary`` populated, or ``error`` set if the
            LLM call fails.
        """
        if state.get("error"):
            logger.debug("Summarize node skipped — upstream error: %s", state["error"])
            return state

        logger.debug("Summarize node — generating summary")
        try:
            summary = generate_summary(state["transcript_content"], llm)
            if not validate_summary_length(summary):
                logger.warning(
                    "Summary is %d chars — exceeds %d char limit",
                    len(summary),
                    CHAR_LIMIT,
                )
            return {**state, "summary": summary}
        except Exception as exc:
            logger.error("Summarize node failed: %s", exc)
            return {**state, "error": str(exc), "summary": ""}

    return _summarize_node


def _save_node(state: SummaryState) -> SummaryState:
    """Pipeline node: write the generated summary to the output file.

    Args:
        state: Current pipeline state containing ``summary`` and ``output_path``.

    Returns:
        Unchanged state on success, or state with ``error`` set if saving fails
        or there is no summary to write.
    """
    if state.get("error"):
        logger.debug("Save node skipped — upstream error: %s", state["error"])
        return state

    if not state.get("summary"):
        msg = "No summary produced — skipping save"
        logger.warning(msg)
        return {**state, "error": msg}

    output_path = Path(state["output_path"])
    logger.debug("Save node — writing to: %s", output_path)
    try:
        save_summary(state["summary"], output_path)
        return state
    except Exception as exc:
        logger.error("Save node failed: %s", exc)
        return {**state, "error": str(exc)}


def build_graph(llm: ChatGroq) -> CompiledStateGraph:
    """Assemble and compile the three-node summarisation pipeline.

    Graph topology::

        START ──► load ──► summarize ──► save ──► END

    Args:
        llm: A configured Groq LLM client injected into the summarize node.

    Returns:
        A compiled :class:`~langgraph.graph.state.CompiledStateGraph` ready
        to call with :meth:`~langgraph.graph.state.CompiledStateGraph.invoke`.
    """
    logger.debug("Building LangGraph summarisation pipeline")
    graph = StateGraph(SummaryState)
    graph.add_node("load", _load_node)
    graph.add_node("summarize", _make_summarize_node(llm))
    graph.add_node("save", _save_node)
    graph.add_edge(START, "load")
    graph.add_edge("load", "summarize")
    graph.add_edge("summarize", "save")
    graph.add_edge("save", END)
    compiled = graph.compile()
    logger.info("LangGraph pipeline compiled successfully")
    return compiled
