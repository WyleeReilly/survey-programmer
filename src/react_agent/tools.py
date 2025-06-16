
from __future__ import annotations

from typing import Any, Callable, List, Optional, cast, Union

from langchain_tavily import TavilySearch  # type: ignore[import-not-found]
from react_agent.configuration import Configuration
import asyncio
# from react_agent.util.file_util import HermesFileUtil
info = """Utility tools used by the LangGraph agent.

Includes:
    • `search` – wrapper around Tavily for general web search
    • `load_survey_file` – fetches a file from Hermes (by file‑ID) and, for now,
      returns the plain‑text contents when the file is a .docx survey document.

Both tools are designed as **async** callables so the graph can await them.
"""
# ---------------------------------------------------------------------------
# Search wrapper (kept from the original template)
# ---------------------------------------------------------------------------
async def search(query: str) -> Optional[dict[str, Any]]:
    """General web search via Tavily.

    Useful for current‑events and broad fact‑finding queries.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


# ---------------------------------------------------------------------------
# NEW: load_survey_file
# ---------------------------------------------------------------------------
async def load_survey_file(file_id: Union[int, str]) -> str:
    """Return the plain‑text contents of a survey file stored in Hermes.

    Parameters
    ----------
    file_id : int | str
        Hermes file identifier (e.g. the integer you get back from the
        `upload_file` endpoint).

    Returns
    -------
    str
        Plain‑text extracted from the .docx survey document.

    Notes
    -----
    • Currently supports **.docx**.  If other MIME types are required,
      extend `HermesFileUtil` with additional parsers (CSV, PPTX, PDF…).
    • Credentials / env are read inside `HermesFileUtil` – fill them in
      once during project setup.
    """
    # HermesFileUtil already caches tokens, so we can reuse one instance
    from react_agent.util.file_util import HermesFileUtil

    hermes = HermesFileUtil()

    # The helper raises if the file isn't .docx or can't be downloaded.
    return await asyncio.to_thread(hermes.read_docx_text, file_id)



# ---------------------------------------------------------------------------
# Expose tools to the agent runtime
# ---------------------------------------------------------------------------
TOOLS: List[Callable[..., Any]] = [search, load_survey_file]
