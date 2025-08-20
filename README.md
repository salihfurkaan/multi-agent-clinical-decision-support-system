# 🏥 Multi-Agent Clinical Decision Support System

A **modular multi-agent system** that leverages **specialized Large Language Models (LLMs)** and **biomedical tools** to support clinical decision-making. This project simulates a collaborative team of AI clinicians, where each agent plays a distinct role (diagnosis, imaging, treatment, pharmacology, patient communication, supervision).

## 🚀 Features

- **Multi-Agent Collaboration**: Specialized agents working together for safe, explainable outputs.
- **Medical Domain LLMs**: Uses cutting-edge medical and general-purpose LLMs (Palmyra-Med, Meditron, LLaMA-3).
- **Evidence Retrieval**: Integration with **PubMed**, **OpenTargets**, and **ClinicalTrials MCP** for up-to-date medical literature.
- **Pharmacology Safety**: Automated checks for drug–drug interactions, dosage appropriateness, and allergies.
- **Explainable Communication**: Patient-friendly summaries generated in natural, spoken language.
- **Supervisor Oversight**: Ensures safe, consistent, and guideline-adherent recommendations.
- **Memory with Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for memory and context tracking.

## 🧑‍⚕️ Agents

| Agent | Role / Function | Chosen Model | Tools |
|-------|-----------------|--------------|-------|
| 🧠 **Diagnosis Agent** | Generates differential diagnoses from patient data | **Palmyra-Med-70B-32K** | — |
| 🔬 **Imaging & Lab Agent** | Suggests tests, interprets lab/imaging results | **Meditron-7B** | — |
| 💊 **Treatment Agent** | Recommends treatment plans and interventions | **Palmyra-Med-70B-32K** | PubMedTool |
| 🧪 **Pharmacology Agent** | Drug interactions, allergies, dosage safety | **Meditron-7B** | OpenTargets, PubMed, ClinicalTrials MCP |
| 🧑‍⚖️ **Supervisor Agent** | Oversees, resolves contradictions, ensures safety | **Palmyra-Med-70B-32K** | PubMedTool |
| 🗣️ **Patient Communication Agent** | Converts outputs into empathetic, patient-friendly speech | **Meta-LLaMA-3-8B-Instruct** | — |

## ⚙️ Setup

### Prerequisites
- Python 3.12
- UV package manager (recommended)
- Git

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/salihfurkaan/multi-agent-clinical-decision-support-system.git
cd multi-agent-clinical-decision-support-system
```

### 2️⃣ Install Dependencies
Using UV (recommended):
```bash
uv pip install -e .
```

Or using pip:
```bash
pip install -e .
```

### 3️⃣ Configure API Keys
Create a `.env` file in the project root with your API keys:
```bash
HF_TOKEN=your_hf_key
```

### 4️⃣ Run the System
```bash
python -m src.multiagent_clinicaldecisionsupport.main
```

## 🐳 Docker Support
Build and run using Docker:
```bash
docker build -t clinical-decision-support .
docker run --env-file .env clinical-decision-support
```

## 🧭 Workflow

The system runs tasks in the following sequence:
1. **Diagnosis Agent** → Suggests differential diagnoses
2. **Imaging & Lab Agent** → Recommends or interprets diagnostic tests
3. **Treatment Agent** → Proposes treatment plan, checks PubMed for evidence
4. **Pharmacology Agent** → Validates medication safety with drug databases
5. **Supervisor Agent** → Integrates outputs, resolves contradictions, ensures safety
6. **Patient Communication Agent** → Generates empathetic, patient-friendly explanation

## 🛠️ Tools Integrated
- **PubMedTool** → Fetch latest biomedical research  (Update the EMAIL constant in pubmed_mcp.py with your email address (required by NCBI))
- **OpenTargets MCP** → Drug-target associations
- **ClinicalTrials MCP** → Clinical trials database

## 📊 Memory
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Purpose**: Maintain patient context, support retrieval-augmented reasoning.

## 🧪 Testing
Run tests using pytest:
```bash
pytest tests/
```

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.

## ⚠️ Disclaimer
This system is for research and educational purposes only. It is not a substitute for professional medical advice. Always consult a licensed clinician before making medical decisions.
