import os

from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task


def run_crewai_demo() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

    researcher = Agent(
        role="Researcher",
        goal="Collect factual context on LlamaIndex and CrewAI",
        backstory=(
            "You are a careful analyst who explains frameworks clearly and accurately."
        ),
        verbose=True,
        llm="gpt-4o-mini",
    )

    writer = Agent(
        role="Writer",
        goal="Turn the research into a concise and useful answer",
        backstory=(
            "You are a strong technical writer who turns complex ideas into readable summaries."
        ),
        verbose=True,
        llm="gpt-4o-mini",
    )

    task1 = Task(
        description=(
            "Research the core purpose of LlamaIndex and CrewAI. Explain how they are different "
            "and how they can complement each other in AI workflows."
        ),
        agent=researcher,
        expected_output="A short but factual summary of each framework and how they fit together.",
    )

    task2 = Task(
        description=(
            "Write a polished final answer for a beginner. Keep it concise, clear, and technical but approachable."
        ),
        agent=writer,
        expected_output="A beginner-friendly explanation in a few paragraphs.",
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    print(result)
