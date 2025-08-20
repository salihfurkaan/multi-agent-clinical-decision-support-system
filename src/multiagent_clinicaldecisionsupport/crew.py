from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from mcp import StdioServerParameters
import os
from crewai import LLM
from crewai_tools import MCPServerAdapter
from multiagent_clinicaldecisionsupport.tools.pubmed import PubMedTool
from src.multiagent_clinicaldecisionsupport.models.transformer_llm import TransformersLLM

model_meditron = TransformersLLM(
    model_name="epfl-llm/meditron-7b"  
)

model_llama = TransformersLLM(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct"
)

model_palmyra = TransformersLLM(
    model_name="Writer/Palmyra-Med-70B-32K"
)

@CrewBase
class MultiagentClinicaldecisionsupport():
    """MultiagentClinicaldecisionsupport crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    mcp_server_params = [StdioServerParameters(
        command="python",
        args=[r"C:\Users\Salih Furkan\OneDrive\Masaüstü\Internship\mcps\clinicaltrialsgov_mcp.py"],
        env={"UV_PYTHON": "3.12", **os.environ},
    ),
    StdioServerParameters(
        command="python",
        args=[r"C:\Users\Salih Furkan\OneDrive\Masaüstü\Internship\mcps\opentargets_mcp.py"],
        env={"UV_PYTHON": "3.12", **os.environ},
    ),
    StdioServerParameters(
        command="python",
        args=[r"C:\Users\Salih Furkan\OneDrive\Masaüstü\Internship\mcps\pubmed_mcp.py"],
        env={"UV_PYTHON": "3.12", **os.environ},
    )]

    @agent
    def diagnostician(self) -> Agent:
        return Agent(
            config=self.agents_config['diagnostician'], 
            reasoning=True,
            max_reasoning_attempts=3,
            respect_context_window=True,
            llm=model_palmyra,
            cache=True,
            max_rpm=10,
            max_iter=15,
            max_retry_limit=2,
            verbose=False
        )

    @agent
    def imaging_and_laboratory_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['imaging_and_laboratory_specialist'], 
            verbose=False,
            multimodal=True,
            reasoning=True,
            max_rpm=10,
            max_reasoning_attempts=3,
            llm=model_meditron,
            cache=True,
            max_iter=12,
            max_retry_limit=2
        )

    @agent
    def clinical_treatment_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['clinical_treatment_specialist'], 
            verbose=False,
            reasoning=True,
            tools= [PubMedTool()],
            respect_context_window=True,
            llm=model_palmyra,
            cache=True,
            max_rpm=10,
            max_iter=12,
            max_retry_limit=2
        )


    @agent
    def pharmacology_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['pharmacology_specialist'], 
            verbose=False,
            reasoning=True,
            tools= self.get_mcp_tools(), #will stay
            respect_context_window=True,
            llm=model_meditron,
            cache=True,
            max_rpm=10,
            max_iter=12,
            max_retry_limit=2
        )

    @agent
    def supervisor(self) -> Agent:
        return Agent(
            config=self.agents_config['supervisor'], 
            verbose=False,
            multimodal=True,
            reasoning=True,
            allow_delegation=True,
            tools= [PubMedTool()],
            respect_context_window=True,
            llm=model_palmyra,
            cache=True,
            max_iter=3,
            max_retry_limit=2,
            max_rpm=8
        )


    @agent
    def patient_communication_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['patient_communication_specialist'], 
            verbose=False,
            reasoning=False,
            respect_context_window=True,
            llm=model_llama,
            cache=True,
            max_iter=3,
            max_retry_limit=2
        )

#---------------------------------TASKS------------------------------------------------

    @task
    def diagnosis_task(self) -> Task:
        return Task(
            config=self.tasks_config['diagnosis_task'], 
        )

    @task
    def imaging_lab_task(self) -> Task:
        return Task(
            config=self.tasks_config['imaging_lab_task'], 
        )

    @task
    def pharmacology_task(self) -> Task:
        return Task(
            config=self.tasks_config['pharmacology_task'], 
        )

    @task
    def treatment_task(self) -> Task:
        return Task(
            config=self.tasks_config['treatment_task'],     
        )

    @task
    def patient_communication_task(self) -> Task:
        return Task(
            config=self.tasks_config['patient_communication_task'], 
        )

#---------------------------------CREW------------------------------------------------

    @crew
    def crew(self) -> Crew:
        """Creates the MultiagentClinicaldecisionsupport crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
            memory=True,
            embedder = {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }
        )
