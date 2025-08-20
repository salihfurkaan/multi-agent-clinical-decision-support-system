#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from multiagent_clinicaldecisionsupport.crew import MultiagentClinicaldecisionsupport

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

#dummy data
inputs = {
        "patient_information": {
            "name": "John Doe",
            "age": 30,
            "gender": "Male"
        },
        "clinical_presentation": {
            "complaint": "Persistent chest pain and shortness of breath",
            "description": "Pain is sharp, worsens with exertion, started 3 days ago",
            "duration": "3 days",
            "additional_notes": "Patient reports no fever or cough"
        },

        "medical_history": "Hypertension, type 2 diabetes",
        "medications_and_allergies": {
            "current_medications": "Metformin, Lisinopril",
            "allergies": "Penicillin"
        },
        "social_history": {
            "smoking": "1 pack a day",
            "alcohol": "Occasional",
            "occupation": "construction worker"
        },
        "family_history": "Father had heart disease, Mother had stroke at 60",
        "attachments": ""
    }

def run():
    """
    Run the crew.
    """
    
    try:
        MultiagentClinicaldecisionsupport().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    try:
        MultiagentClinicaldecisionsupport().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MultiagentClinicaldecisionsupport().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """    
    try:
        MultiagentClinicaldecisionsupport().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
