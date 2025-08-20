from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from langchain_community.tools.pubmed.tool import PubmedQueryRun

pubmed_tool = PubmedQueryRun()

class PubMedTool(BaseTool):
    name: str = "PubMed Tool"
    description: str = (
        "This tool allows you to search for PubMed articles."
    )

    def _run(self, argument: str) -> str:
        """Runs the PubMed tool."""
        try:
            return pubmed_tool.invoke(argument)
        except Exception as e:
            return f"An error occurred while running the PubMed tool: {e}"
