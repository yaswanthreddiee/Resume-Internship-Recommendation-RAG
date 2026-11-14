"""
Dataset generator for Resume-Based Internship Recommendation System.
Generates:
1. data/internships.json: 60 diverse, realistic internship postings across 10 categories.
2. data/test_candidates.json: 10 benchmark candidates + 10 edge case test profiles.
3. data/resumes/: Individual candidate JSON files.

All synthetic data is explicitly labelled as synthetic benchmark records.
"""

import json
import os
from typing import Dict, List, Any

INTERNSHIP_CATEGORIES = [
    "Artificial Intelligence",
    "Machine Learning",
    "Data Science",
    "Python Development",
    "Backend Development",
    "Frontend Development",
    "Full Stack Development",
    "Generative AI",
    "Java Development",
    "Data Analytics"
]

def build_internships_dataset() -> List[Dict[str, Any]]:
    internships = [
        # 1. Artificial Intelligence (6 records)
        {
            "internship_id": "INT-AI-001",
            "title": "AI Research & Computer Vision Intern",
            "company_name": "Cognitive Vision Labs [Synthetic Benchmark]",
            "category": "Artificial Intelligence",
            "description": "Assist in researching and fine-tuning state-of-the-art vision models (YOLO, ViT) for automated defect inspection and object recognition.",
            "required_skills": ["Python", "PyTorch", "Computer Vision", "OpenCV", "Deep Learning"],
            "preferred_skills": ["CUDA", "TensorRT", "Docker", "Git"],
            "education_requirements": "B.Tech/M.Tech in CS, AI, Robotics or related quantitative field (Graduation: 2024-2026)",
            "experience_requirements": "Hands-on academic projects or coursework in Deep Learning & Vision",
            "location": "Bengaluru, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Strong foundation in linear algebra, multivariable calculus, and PyTorch tensors.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-AI-002",
            "title": "Applied AI Engineering Intern",
            "company_name": "Synthetica AI [Synthetic Benchmark]",
            "category": "Artificial Intelligence",
            "description": "Develop and deploy neural network inference pipelines and optimize model latency for edge and cloud deployment.",
            "required_skills": ["Python", "TensorFlow", "PyTorch", "REST APIs", "Model Optimization"],
            "preferred_skills": ["ONNX", "Docker", "FastAPI", "Kubernetes"],
            "education_requirements": "B.E./B.Tech in Computer Science, Data Science, or ECE",
            "experience_requirements": "Prior implementation of convolutional or transformer architectures in college projects",
            "location": "Hyderabad, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Familiarity with profiling inference latency and memory footprints.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-AI-003",
            "title": "Speech & Audio AI Intern",
            "company_name": "SonicWave AI [Synthetic Benchmark]",
            "category": "Artificial Intelligence",
            "description": "Train and evaluate automatic speech recognition (ASR) and audio synthesis pipelines using wav2vec and Whisper models.",
            "required_skills": ["Python", "PyTorch", "Librosa", "Audio Processing", "Deep Learning"],
            "preferred_skills": ["HuggingFace", "FastAPI", "Linux"],
            "education_requirements": "B.Tech/M.Tech/MS in Computer Science or Electrical Engineering",
            "experience_requirements": "Coursework in DSP or audio processing project experience",
            "location": "Pune, India",
            "work_mode": "Remote",
            "duration": "3 months",
            "other_requirements": "Understanding of spectrograms, Mel-frequency cepstral coefficients, and CTC loss.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-AI-004",
            "title": "Reinforcement Learning & Simulation Intern",
            "company_name": "Kinetic Autonomy [Synthetic Benchmark]",
            "category": "Artificial Intelligence",
            "description": "Implement RL algorithms (PPO, SAC, DQN) in physics simulators (Gymnasium, Isaac Sim) for robotic arm trajectory planning.",
            "required_skills": ["Python", "PyTorch", "Reinforcement Learning", "Gymnasium", "NumPy"],
            "preferred_skills": ["ROS", "C++", "MuJoCo", "Git"],
            "education_requirements": "Degree in Robotics, Mechanical, Computer Science, or Mechatronics",
            "experience_requirements": "Simulation project in RL or robotics control",
            "location": "Bengaluru, India",
            "work_mode": "On-site",
            "duration": "6 months",
            "other_requirements": "Strong mathematical modeling skills in kinematics and Markov Decision Processes.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-AI-005",
            "title": "Medical Imaging AI Intern",
            "company_name": "HealthVision Biometrics [Synthetic Benchmark]",
            "category": "Artificial Intelligence",
            "description": "Develop 3D medical image segmentation models (UNet, nnU-Net) on CT and MRI DICOM datasets with strict validation protocols.",
            "required_skills": ["Python", "PyTorch", "Medical Imaging", "SimpleITK", "Deep Learning"],
            "preferred_skills": ["MONAI", "Docker", "Weights & Biases"],
            "education_requirements": "B.Tech/M.Tech in Biomedical Engineering, AI, or Computer Science",
            "experience_requirements": "Familiarity with medical dataset handling and volumetric segmentation",
            "location": "Chennai, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Knowledge of sensitivity, specificity, and Dice coefficient metrics.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-AI-006",
            "title": "AI Ethics & Model Fairness Intern",
            "company_name": "Veritas AI Systems [Synthetic Benchmark]",
            "category": "Artificial Intelligence",
            "description": "Benchmark algorithmic fairness, explainability (SHAP, LIME), and robustness of deployed neural networks against adversarial attacks.",
            "required_skills": ["Python", "Scikit-Learn", "PyTorch", "Model Explainability", "Statistics"],
            "preferred_skills": ["SHAP", "LIME", "Pandas", "Matplotlib"],
            "education_requirements": "Degree in Data Science, Computer Science, or Computational Social Science",
            "experience_requirements": "Academic research or mini-projects analyzing dataset bias",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "3 months",
            "other_requirements": "Strong analytical writing and empirical evaluation mindset.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },

        # 2. Machine Learning (6 records)
        {
            "internship_id": "INT-ML-001",
            "title": "Machine Learning Engineer Intern",
            "company_name": "Predictive Matrix Labs [Synthetic Benchmark]",
            "category": "Machine Learning",
            "description": "Build end-to-end predictive pipelines using gradient boosting and ensemble models for customer churn and demand forecasting.",
            "required_skills": ["Python", "Scikit-Learn", "XGBoost", "Pandas", "Feature Engineering"],
            "preferred_skills": ["LightGBM", "MLflow", "SQL", "Git"],
            "education_requirements": "B.E./B.Tech/MCA in Computer Science or Information Technology",
            "experience_requirements": "Hands-on projects with tabular datasets, cross-validation, and hyperparameter tuning",
            "location": "Pune, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Deep understanding of ROC-AUC, precision-recall trade-offs, and data leakage prevention.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-ML-002",
            "title": "MLOps & Pipeline Engineering Intern",
            "company_name": "PipelineOps Tech [Synthetic Benchmark]",
            "category": "Machine Learning",
            "description": "Automate continuous training pipelines, model registry versioning, and drift monitoring using MLflow and DVC.",
            "required_skills": ["Python", "Docker", "MLflow", "Scikit-Learn", "CI/CD"],
            "preferred_skills": ["DVC", "Airflow", "FastAPI", "Linux"],
            "education_requirements": "B.Tech in CS/IT or Software Engineering",
            "experience_requirements": "Experience building and containerizing Python ML scripts",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Understanding of automated testing, reproducible seed setting, and artifact tracking.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-ML-003",
            "title": "Time Series Forecasting Intern",
            "company_name": "QuantChrono Analytics [Synthetic Benchmark]",
            "category": "Machine Learning",
            "description": "Develop financial and operational time series forecasting models using Prophet, ARIMA, and temporal neural networks.",
            "required_skills": ["Python", "Pandas", "Statsmodels", "Time Series Analysis", "Scikit-Learn"],
            "preferred_skills": ["Prophet", "PyTorch", "Seaborn", "SQL"],
            "education_requirements": "B.Sc/M.Sc/B.Tech in Mathematics, Statistics, Economics, or Computer Science",
            "experience_requirements": "Projects handling stationarity, seasonality, and rolling forecast cross-validation",
            "location": "Mumbai, India",
            "work_mode": "Hybrid",
            "duration": "3 months",
            "other_requirements": "Solid grasp of autocorrelation, cointegration, and MAPE/RMSE metrics.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-ML-004",
            "title": "Recommendation Systems Intern",
            "company_name": "EngageRecs Platform [Synthetic Benchmark]",
            "category": "Machine Learning",
            "description": "Construct collaborative filtering, matrix factorization, and two-tower neural recommendation engines for digital content.",
            "required_skills": ["Python", "Scikit-Learn", "PyTorch", "Collaborative Filtering", "NumPy"],
            "preferred_skills": ["Implicit", "FastAPI", "Redis", "SQL"],
            "education_requirements": "B.Tech/M.Tech in CS or Information Science",
            "experience_requirements": "Experience building recommendation or ranking models on Kaggle/academic benchmarks",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Knowledge of NDCG, Recall@K, and cold-start mitigations.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-ML-005",
            "title": "Deep Learning Computer Vision Intern",
            "company_name": "Visionary AI Systems [Synthetic Benchmark]",
            "category": "Machine Learning",
            "description": "Train and evaluate deep convolutional networks and vision transformers for multi-class classification and instance segmentation.",
            "required_skills": ["Python", "TensorFlow", "Keras", "Deep Learning", "OpenCV"],
            "preferred_skills": ["PyTorch", "Albumentations", "Docker"],
            "education_requirements": "B.Tech/B.E. in Computer Science or Artificial Intelligence",
            "experience_requirements": "Coursework projects using CNNs (ResNet, EfficientNet) on image datasets",
            "location": "Noida, India",
            "work_mode": "On-site",
            "duration": "6 months",
            "other_requirements": "Strong data augmentation and GPU transfer learning workflow knowledge.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-ML-006",
            "title": "Fraud Detection ML Intern",
            "company_name": "FinShield Risk Labs [Synthetic Benchmark]",
            "category": "Machine Learning",
            "description": "Design anomaly detection and highly imbalanced classification models (Isolation Forests, SMOTE, CatBoost) for payment transactions.",
            "required_skills": ["Python", "Scikit-Learn", "CatBoost", "Anomaly Detection", "Pandas"],
            "preferred_skills": ["Imbalanced-Learn", "SQL", "FastAPI"],
            "education_requirements": "Degree in Data Science, Statistics, or Computer Science",
            "experience_requirements": "Mini-projects working with skewed class distributions and PR curves",
            "location": "Gurugram, India",
            "work_mode": "Hybrid",
            "duration": "4 months",
            "other_requirements": "Familiarity with financial compliance considerations and cost-sensitive learning.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },

        # 3. Data Science (6 records)
        {
            "internship_id": "INT-DS-001",
            "title": "Data Science Research Intern",
            "company_name": "InsightSphere Data Labs [Synthetic Benchmark]",
            "category": "Data Science",
            "description": "Formulate hypothesis tests, conduct exploratory data analysis on multi-terabyte datasets, and build interpretable regression/classification models.",
            "required_skills": ["Python", "SQL", "Pandas", "Statistics", "Hypothesis Testing"],
            "preferred_skills": ["Scikit-Learn", "R", "Tableau", "Git"],
            "education_requirements": "B.Sc/M.Sc/B.Tech in Statistics, Mathematics, Data Science, or Economics",
            "experience_requirements": "Strong university background in experimental design, p-values, ANOVA, and linear regression",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Ability to synthesize statistical findings into actionable executive summaries.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-DS-002",
            "title": "Quantitative Data Scientist Intern",
            "company_name": "AlphaMetrics Capital [Synthetic Benchmark]",
            "category": "Data Science",
            "description": "Analyze market microstructure datasets, backtest statistical arbitrage factors, and simulate risk distributions.",
            "required_skills": ["Python", "NumPy", "Pandas", "Probability & Statistics", "SciPy"],
            "preferred_skills": ["SQL", "Matplotlib", "Financial Modeling"],
            "education_requirements": "B.Tech/Dual Degree in CS, Math, Engineering or Financial Engineering",
            "experience_requirements": "Strong quantitative coursework and algorithmic problem-solving",
            "location": "Mumbai, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Deep appreciation for stochastic processes and robust statistical estimators.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-DS-003",
            "title": "Healthcare Data Science Intern",
            "company_name": "BioHealth Analytics [Synthetic Benchmark]",
            "category": "Data Science",
            "description": "Apply survival analysis, longitudinal regression, and cohort clustering to anonymized clinical trial datasets.",
            "required_skills": ["Python", "Pandas", "Scikit-Learn", "Survival Analysis", "SQL"],
            "preferred_skills": ["Lifelines", "Seaborn", "JupyterLab"],
            "education_requirements": "Degree in Bioinformatics, Data Science, Biostatistics, or Computer Science",
            "experience_requirements": "Academic projects with biomedical data or cohort studies",
            "location": "Hyderabad, India",
            "work_mode": "Remote",
            "duration": "4 months",
            "other_requirements": "Rigorous attention to data sanitation, missing value imputation, and clinical validity.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-DS-004",
            "title": "Customer Intelligence Data Scientist Intern",
            "company_name": "OmniRetail Insights [Synthetic Benchmark]",
            "category": "Data Science",
            "description": "Execute RFM segmentation, lifetime value (LTV) estimation, and market basket association analysis on retail transaction logs.",
            "required_skills": ["Python", "SQL", "Pandas", "K-Means Clustering", "Exploratory Data Analysis"],
            "preferred_skills": ["PowerBI", "Scikit-Learn", "Mlxtend"],
            "education_requirements": "B.Tech/BBA-Data Analytics/MCA/MBA-Analytics",
            "experience_requirements": "Coursework or capstone in customer behavior modeling or retail data",
            "location": "Gurugram, India",
            "work_mode": "Hybrid",
            "duration": "3 months",
            "other_requirements": "Demonstrated flair for cohort retention charts and silhouette cluster scoring.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-DS-005",
            "title": "A/B Testing & Causal Inference Intern",
            "company_name": "Experimentation Engine [Synthetic Benchmark]",
            "category": "Data Science",
            "description": "Formulate statistical power calculations, sample size sizing, and variance reduction techniques (CUPED) for product experiments.",
            "required_skills": ["Python", "Statistics", "A/B Testing", "SQL", "SciPy"],
            "preferred_skills": ["Statsmodels", "Pandas", "CausalML"],
            "education_requirements": "B.S./M.S. in Statistics, Economics, Applied Math or CS",
            "experience_requirements": "Knowledge of central limit theorem, bootstrap estimation, and hypothesis tests",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Strict adherence to scientific integrity and false positive rate controls.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-DS-006",
            "title": "Climate & Geo-Data Science Intern",
            "company_name": "TerraSensor Geospatial [Synthetic Benchmark]",
            "category": "Data Science",
            "description": "Process remote sensing satellite raster grids and environmental sensor time series to model regional drought indices.",
            "required_skills": ["Python", "Pandas", "Geopandas", "Raster Data", "Statistics"],
            "preferred_skills": ["Xarray", "Matplotlib", "Scikit-Learn"],
            "education_requirements": "Degree in Environmental Science, Geoinformatics, or Computer Science",
            "experience_requirements": "Familiarity with spatial coordinates, NetCDF/GeoTIFF formats, and Python",
            "location": "Dehradun, India",
            "work_mode": "Remote",
            "duration": "3 months",
            "other_requirements": "Keen interest in climate change impact modeling.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },

        # 4. Python Development (6 records)
        {
            "internship_id": "INT-PY-001",
            "title": "Python Core & Automation Engineer Intern",
            "company_name": "PyCore Automations [Synthetic Benchmark]",
            "category": "Python Development",
            "description": "Develop robust Python automation scripts, ETL data extraction routines, and internal developer tooling utilizing asyncio and multiprocessing.",
            "required_skills": ["Python", "Object-Oriented Programming", "Asyncio", "APIs", "Git"],
            "preferred_skills": ["Pytest", "Requests", "Docker", "Linux"],
            "education_requirements": "B.E./B.Tech in Computer Science, Information Science, or MCA (2024-2026)",
            "experience_requirements": "Clean GitHub portfolio demonstrating idiomatic Python scripts and modular packaging",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Knowledge of Python data structures, generators, decorators, and context managers.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-PY-002",
            "title": "Python Web Scraping & Data Pipeline Intern",
            "company_name": "ScrapeFlow Intelligence [Synthetic Benchmark]",
            "category": "Python Development",
            "description": "Architect distributed web crawlers using Scrapy and Playwright to harvest, sanitize, and persist structured web data into document stores.",
            "required_skills": ["Python", "Scrapy", "BeautifulSoup", "Selenium", "XPath"],
            "preferred_skills": ["Playwright", "MongoDB", "Asyncio", "Docker"],
            "education_requirements": "B.Tech/BCA/MCA in Computer Science or related degree",
            "experience_requirements": "Experience dealing with CAPTCHAs, proxy rotation, and asynchronous page loads",
            "location": "Noida, India",
            "work_mode": "Hybrid",
            "duration": "3 months",
            "other_requirements": "Respect for robots.txt and ethical rate-limiting policies.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-PY-003",
            "title": "Python CLI & Developer Tooling Intern",
            "company_name": "DevEx Toolsmiths [Synthetic Benchmark]",
            "category": "Python Development",
            "description": "Construct high-ergonomic Command Line Interfaces using Click, Typer, and Rich for cloud resource provisioning and local builds.",
            "required_skills": ["Python", "Click", "Typer", "Subprocess", "Unit Testing"],
            "preferred_skills": ["Rich", "Packaging (Poetry/Flit)", "Bash", "Git"],
            "education_requirements": "B.Tech in Computer Science or open-source contributor",
            "experience_requirements": "Demonstrated CLI tools or utilities built in Python",
            "location": "Pune, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Obsession with clean terminal UX, cross-platform compatibility, and graceful signal exits.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-PY-004",
            "title": "Python Data Engineering Intern",
            "company_name": "DataNexus Pipelines [Synthetic Benchmark]",
            "category": "Python Development",
            "description": "Build high-throughput batch and streaming ingest pipelines using Python, PySpark, and Kafka consumers.",
            "required_skills": ["Python", "SQL", "Pandas", "ETL Pipelines", "Data Ingestion"],
            "preferred_skills": ["PySpark", "Kafka", "PostgreSQL", "Docker"],
            "education_requirements": "B.E./B.Tech/M.Sc in CS, IT, or Big Data Systems",
            "experience_requirements": "Projects handling database migrations, schema validation, and CSV/JSON parsing",
            "location": "Chennai, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Understanding of data warehouse idempotency and parquet file storage.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-PY-005",
            "title": "Python Test Automation & QA Intern",
            "company_name": "QualityFirst Systems [Synthetic Benchmark]",
            "category": "Python Development",
            "description": "Write automated test suites, fixture mockings, and contract verification tests using Pytest and Allure reports.",
            "required_skills": ["Python", "Pytest", "Automation Testing", "API Testing", "Git"],
            "preferred_skills": ["Mocking/Unittest", "Requests", "Jenkins", "Docker"],
            "education_requirements": "B.Tech/MCA in Computer Science",
            "experience_requirements": "Hands-on experience testing REST endpoints and writing unit tests",
            "location": "Kolkata, India",
            "work_mode": "Remote",
            "duration": "4 months",
            "other_requirements": "Understanding of boundary testing, parameterization, and regression test suites.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-PY-006",
            "title": "Python Distributed Systems Intern",
            "company_name": "ScaleGrid Computing [Synthetic Benchmark]",
            "category": "Python Development",
            "description": "Implement task workers, distributed locks, and retry queues using Celery, Redis, and Python concurrency primitives.",
            "required_skills": ["Python", "Celery", "Redis", "Multithreading", "Data Structures"],
            "preferred_skills": ["RabbitMQ", "Docker", "Linux System Calls"],
            "education_requirements": "B.Tech in Computer Science or Software Engineering",
            "experience_requirements": "Coursework in Distributed Systems, OS concepts, and Socket programming",
            "location": "Bengaluru, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Proficiency in debugging thread race conditions and memory leaks.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },

        # 5. Backend Development (6 records)
        {
            "internship_id": "INT-BE-001",
            "title": "Python Backend Engineer Intern (Django / REST)",
            "company_name": "Apex Cloud Services [Synthetic Benchmark]",
            "category": "Backend Development",
            "description": "Develop modular RESTful API services, manage relational databases (PostgreSQL), and implement authentication protocols with Django REST Framework.",
            "required_skills": ["Python", "Django", "Django REST Framework", "PostgreSQL", "REST APIs"],
            "preferred_skills": ["JWT", "Redis", "Docker", "Pytest"],
            "education_requirements": "B.Tech / B.E. / MCA in Computer Science or Information Technology",
            "experience_requirements": "Academic or freelance backend development with Django and relational databases",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Understanding of ORM optimizations, indexing, foreign keys, and middleware.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-BE-002",
            "title": "FastAPI Microservices Intern",
            "company_name": "Velox Microservices [Synthetic Benchmark]",
            "category": "Backend Development",
            "description": "Design ultra-low latency asynchronous microservices using FastAPI, Pydantic schemas, and SQLAlchemy ORM.",
            "required_skills": ["Python", "FastAPI", "Pydantic", "SQLAlchemy", "Asyncio"],
            "preferred_skills": ["PostgreSQL", "Alembic", "Docker", "Swagger/OpenAPI"],
            "education_requirements": "B.Tech in CS or Software Engineering",
            "experience_requirements": "Creation of asynchronous backend endpoints and schema validation projects",
            "location": "Hyderabad, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Knowledge of dependency injection patterns and OpenAPI spec generation.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-BE-003",
            "title": "Flask Backend & Cloud Integrations Intern",
            "company_name": "CloudNimbus Solutions [Synthetic Benchmark]",
            "category": "Backend Development",
            "description": "Maintain lightweight microservices in Flask, interface with AWS S3 / DynamoDB, and construct secure webhooks.",
            "required_skills": ["Python", "Flask", "SQL", "REST APIs", "AWS"],
            "preferred_skills": ["Boto3", "Docker", "Gunicorn", "Git"],
            "education_requirements": "B.E./B.Tech/MCA in Computer Science or IT",
            "experience_requirements": "Mini-projects deploying Flask apps to cloud environments",
            "location": "Pune, India",
            "work_mode": "Remote",
            "duration": "3 months",
            "other_requirements": "Solid security principles regarding CORS, CSRF, and SQL Injection prevention.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-BE-004",
            "title": "Node.js & Express Backend Intern",
            "company_name": "NodeCraft Server Systems [Synthetic Benchmark]",
            "category": "Backend Development",
            "description": "Build high-throughput event-driven microservices using Node.js, Express, MongoDB, and Redis caching.",
            "required_skills": ["JavaScript", "Node.js", "Express.js", "MongoDB", "REST APIs"],
            "preferred_skills": ["TypeScript", "Redis", "Mongoose", "Jest"],
            "education_requirements": "B.Tech/BCA/MCA in Computer Science",
            "experience_requirements": "Academic projects building REST backends with Express and NoSQL databases",
            "location": "Bengaluru, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Deep understanding of Node.js event loop, promises, and error handling middleware.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-BE-005",
            "title": "Go / Golang Backend Systems Intern",
            "company_name": "GopherScale Infrastructure [Synthetic Benchmark]",
            "category": "Backend Development",
            "description": "Develop high-concurrency microservices, gRPC protocol communication, and rate limiters in Golang.",
            "required_skills": ["Go", "Golang", "REST APIs", "SQL", "Concurrency"],
            "preferred_skills": ["gRPC", "Protobuf", "Docker", "PostgreSQL"],
            "education_requirements": "B.Tech/B.E. in Computer Science",
            "experience_requirements": "Basic personal or college projects demonstrating goroutines, channels, and structs in Go",
            "location": "Hyderabad, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Strong foundation in operating systems, pointers, and memory layout.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-BE-006",
            "title": "Database & Query Optimization Backend Intern",
            "company_name": "DataScale Database Labs [Synthetic Benchmark]",
            "category": "Backend Development",
            "description": "Perform query tuning, index restructuring, schema normalization, and connection pool management across SQL clusters.",
            "required_skills": ["SQL", "PostgreSQL", "Database Design", "Python", "Performance Tuning"],
            "preferred_skills": ["MySQL", "Redis", "EXPLAIN ANALYZE", "Linux"],
            "education_requirements": "B.Tech/M.Tech in CS/IT",
            "experience_requirements": "Deep college project experience with complex relational schemas and joins",
            "location": "Chennai, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Proficiency in transactions, ACID guarantees, and dead-lock diagnosis.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },

        # 6. Frontend Development (6 records)
        {
            "internship_id": "INT-FE-001",
            "title": "React.js Frontend Developer Intern",
            "company_name": "PixelCraft UI Studio [Synthetic Benchmark]",
            "category": "Frontend Development",
            "description": "Build responsive, accessible, and high-performance single page web applications using React 18, TypeScript, and modern CSS frameworks.",
            "required_skills": ["React", "JavaScript", "HTML5", "CSS3", "Responsive Design"],
            "preferred_skills": ["TypeScript", "Tailwind CSS", "Redux Toolkit", "Next.js"],
            "education_requirements": "B.Tech/BCA/MCA in Computer Science or self-taught with exceptional portfolio",
            "experience_requirements": "Demonstrated responsive web applications published on GitHub or Vercel",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Thorough knowledge of React hooks (useState, useEffect, useMemo) and virtual DOM.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-FE-002",
            "title": "Modern JavaScript & Vue.js Intern",
            "company_name": "VueStream Interfaces [Synthetic Benchmark]",
            "category": "Frontend Development",
            "description": "Develop intuitive enterprise management dashboards using Vue 3 Composition API, Pinia state store, and Vite.",
            "required_skills": ["JavaScript", "Vue.js", "HTML5", "CSS3", "REST API Integration"],
            "preferred_skills": ["Pinia", "TypeScript", "Vite", "Tailwind CSS"],
            "education_requirements": "B.E./B.Tech/MCA in CS/IT",
            "experience_requirements": "Frontend projects integrating client-side routes and API endpoints",
            "location": "Pune, India",
            "work_mode": "Hybrid",
            "duration": "3 months",
            "other_requirements": "Understanding of component lifecycles, props, emits, and reactivity.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-FE-003",
            "title": "Next.js & Frontend Performance Intern",
            "company_name": "Apex Web Experience [Synthetic Benchmark]",
            "category": "Frontend Development",
            "description": "Optimize Core Web Vitals, implement Server-Side Rendering (SSR), and craft engaging landing pages using Next.js and Tailwind.",
            "required_skills": ["React", "Next.js", "JavaScript", "CSS3", "Web Performance"],
            "preferred_skills": ["TypeScript", "Framer Motion", "SEO", "Vercel"],
            "education_requirements": "B.Tech in Computer Science or Design",
            "experience_requirements": "Portfolio website built with Next.js or React",
            "location": "Mumbai, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Knowledge of image optimization, font caching, and cumulative layout shift.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-FE-004",
            "title": "Angular Enterprise Frontend Intern",
            "company_name": "ScaleFront Enterprise [Synthetic Benchmark]",
            "category": "Frontend Development",
            "description": "Implement robust enterprise forms, RxJS data streams, and hierarchical routing in Angular 16+.",
            "required_skills": ["Angular", "TypeScript", "JavaScript", "HTML5", "CSS3"],
            "preferred_skills": ["RxJS", "Angular Material", "Karma/Jasmine", "Bootstrap"],
            "education_requirements": "B.Tech/MCA in Computer Science",
            "experience_requirements": "Coursework or projects in Angular and TypeScript",
            "location": "Gurugram, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Strong grip on observables, pipes, services, and two-way binding.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-FE-005",
            "title": "UI/UX Design Systems & Frontend Intern",
            "company_name": "DesignSystem Core [Synthetic Benchmark]",
            "category": "Frontend Development",
            "description": "Bridge the gap between Figma mockups and code by maintaining an accessible React design system library and Storybook docs.",
            "required_skills": ["HTML5", "CSS3", "JavaScript", "React", "Figma to Code"],
            "preferred_skills": ["Storybook", "Tailwind CSS", "Accessibility (WCAG)", "TypeScript"],
            "education_requirements": "Degree in Computer Science, HCI, or Interaction Design",
            "experience_requirements": "Portfolio demonstrating fidelity between Figma designs and coded UI components",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "4 months",
            "other_requirements": "Understanding of semantic HTML, ARIA attributes, and color contrast tokens.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-FE-006",
            "title": "Mobile-Web & Progressive Web App (PWA) Intern",
            "company_name": "MobilityFrontier [Synthetic Benchmark]",
            "category": "Frontend Development",
            "description": "Construct offline-first Progressive Web Applications with Service Workers, indexedDB caching, and touch gestures.",
            "required_skills": ["JavaScript", "HTML5", "CSS3", "React", "Service Workers"],
            "preferred_skills": ["PWA", "IndexedDB", "TypeScript", "Lighthouse"],
            "education_requirements": "B.Tech/BCA in Computer Science",
            "experience_requirements": "Experience creating responsive mobile web experiences",
            "location": "Noida, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Knowledge of manifest.json, cache storage API, and background sync.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },

        # 7. Full Stack Development (6 records)
        {
            "internship_id": "INT-FS-001",
            "title": "Python Full Stack Developer Intern (Django + React)",
            "company_name": "OmniStack Cloud Labs [Synthetic Benchmark]",
            "category": "Full Stack Development",
            "description": "Design and build end-to-end cloud platforms by crafting interactive React interfaces and robust Django REST backends with PostgreSQL.",
            "required_skills": ["Python", "Django", "React", "JavaScript", "PostgreSQL", "REST APIs"],
            "preferred_skills": ["Tailwind CSS", "Docker", "Git", "Redux"],
            "education_requirements": "B.Tech / B.E. in CS/IT or MCA",
            "experience_requirements": "Full-stack projects combining Python backends with modern JavaScript frontends",
            "location": "Bengaluru, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Proficiency in CORS configuration, JWT auth cycles, and end-to-end data flow.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-FS-002",
            "title": "MERN Stack Developer Intern",
            "company_name": "MERNForge Technologies [Synthetic Benchmark]",
            "category": "Full Stack Development",
            "description": "Develop scalable SaaS web apps utilizing MongoDB, Express.js, React, and Node.js with secure authentication and real-time sockets.",
            "required_skills": ["MongoDB", "Express.js", "React", "Node.js", "JavaScript"],
            "preferred_skills": ["Socket.io", "TypeScript", "Tailwind CSS", "JWT"],
            "education_requirements": "B.Tech/MCA/BCA in Computer Science",
            "experience_requirements": "Demonstrated MERN application deployed on cloud platforms",
            "location": "Hyderabad, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Understanding of asynchronous server handlers and React lifecycle state management.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-FS-003",
            "title": "FastAPI + Vue.js Full Stack Intern",
            "company_name": "AgileStack Labs [Synthetic Benchmark]",
            "category": "Full Stack Development",
            "description": "Construct high-performance enterprise dashboards with asynchronous Python FastAPI backend services and reactive Vue 3 components.",
            "required_skills": ["Python", "FastAPI", "Vue.js", "JavaScript", "SQL"],
            "preferred_skills": ["SQLAlchemy", "Pinia", "Docker", "Vite"],
            "education_requirements": "B.Tech/B.E. in Computer Science or Information Technology",
            "experience_requirements": "Projects integrating Vue/React with Python backends",
            "location": "Pune, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Familiarity with Swagger documentation and WebSocket streaming.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-FS-004",
            "title": "Next.js Full Stack & Serverless Intern",
            "company_name": "Veloce Web Systems [Synthetic Benchmark]",
            "category": "Full Stack Development",
            "description": "Build modern full-stack web applications using Next.js App Router, Server Actions, Prisma ORM, and Supabase / PostgreSQL.",
            "required_skills": ["Next.js", "React", "TypeScript", "Node.js", "SQL"],
            "preferred_skills": ["Prisma", "PostgreSQL", "Tailwind CSS", "Supabase"],
            "education_requirements": "B.Tech/BCA/MCA in Computer Science",
            "experience_requirements": "Full-stack projects leveraging Next.js server components and databases",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Knowledge of serverless function cold-starts, caching tags, and revalidation.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-FS-005",
            "title": "Flask + React Full Stack Intern",
            "company_name": "LiteStack Innovations [Synthetic Benchmark]",
            "category": "Full Stack Development",
            "description": "Develop customer-facing web portals with Flask lightweight backend microservices and responsive React components.",
            "required_skills": ["Python", "Flask", "React", "JavaScript", "HTML5"],
            "preferred_skills": ["SQLite/PostgreSQL", "Bootstrap", "Git", "REST APIs"],
            "education_requirements": "B.Tech/B.E. in CS or IT",
            "experience_requirements": "Academic projects connecting Flask endpoints with a React or static frontend",
            "location": "Noida, India",
            "work_mode": "Hybrid",
            "duration": "3 months",
            "other_requirements": "Understanding of HTTP methods, response headers, and cookie handling.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-FS-006",
            "title": "Java Spring Boot + Angular Full Stack Intern",
            "company_name": "GlobalEnterprise Solutions [Synthetic Benchmark]",
            "category": "Full Stack Development",
            "description": "Implement enterprise banking portal features utilizing Java 17, Spring Boot microservices, Hibernate, and Angular frontends.",
            "required_skills": ["Java", "Spring Boot", "Angular", "TypeScript", "SQL"],
            "preferred_skills": ["Hibernate", "PostgreSQL", "Maven", "Docker"],
            "education_requirements": "B.Tech/B.E./MCA in Computer Science",
            "experience_requirements": "Class projects or internships involving Java and frontend components",
            "location": "Chennai, India",
            "work_mode": "On-site",
            "duration": "6 months",
            "other_requirements": "Understanding of MVC architecture, dependency injection, and Angular services.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },

        # 8. Generative AI (6 records)
        {
            "internship_id": "INT-GENAI-001",
            "title": "Generative AI & LLM Systems Intern",
            "company_name": "Nexus Frontier Generative AI [Synthetic Benchmark]",
            "category": "Generative AI",
            "description": "Design and benchmark Retrieval-Augmented Generation (RAG) pipelines, semantic vector search systems, and fine-tune open weights LLMs.",
            "required_skills": ["Python", "Natural Language Processing", "Large Language Models", "LangChain", "Vector Databases"],
            "preferred_skills": ["ChromaDB", "LlamaIndex", "HuggingFace", "Prompt Engineering"],
            "education_requirements": "B.Tech / M.Tech in Computer Science, AI, or Data Science (2024-2026)",
            "experience_requirements": "Hands-on projects implementing RAG, embeddings, or agentic frameworks",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Deep understanding of chunking strategies, cosine similarity, reranking, and hallucination reduction.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-GENAI-002",
            "title": "RAG & Vector Search Engineering Intern",
            "company_name": "SemanticRetriever Labs [Synthetic Benchmark]",
            "category": "Generative AI",
            "description": "Build enterprise search engines utilizing dense embeddings, hybrid keyword-vector retrieval, and reranking cross-encoders.",
            "required_skills": ["Python", "ChromaDB", "Sentence Transformers", "RAG", "Embeddings"],
            "preferred_skills": ["BM25", "Cross-Encoder", "FastAPI", "Docker"],
            "education_requirements": "B.Tech in CS/IT or Artificial Intelligence",
            "experience_requirements": "Demonstrated project building semantic retrieval or document Q&A systems",
            "location": "Hyderabad, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Understanding of vector indexing (HNSW), cosine distance, and precision-recall in retrieval.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-GENAI-003",
            "title": "LLM Fine-Tuning & Prompt Engineering Intern",
            "company_name": "TuringPrompt Labs [Synthetic Benchmark]",
            "category": "Generative AI",
            "description": "Perform parameter-efficient fine-tuning (LoRA, QLoRA) on open-source foundation models (Llama 3, Mistral) and evaluate chain-of-thought prompts.",
            "required_skills": ["Python", "PyTorch", "HuggingFace", "LoRA", "Prompt Engineering"],
            "preferred_skills": ["PEFT", "TRL", "Weights & Biases", "CUDA"],
            "education_requirements": "B.Tech/M.Tech in Computer Science or Data Science",
            "experience_requirements": "Hands-on fine-tuning or quantization projects on HuggingFace Hub",
            "location": "Pune, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Familiarity with attention mechanisms, tokenizers, and loss curves.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-GENAI-004",
            "title": "Multimodal Generative AI Intern",
            "company_name": "VisionaryGenerative [Synthetic Benchmark]",
            "category": "Generative AI",
            "description": "Experiment with image diffusion models (Stable Diffusion, ControlNet) and vision-language models (CLIP, LLaVA) for creative media pipelines.",
            "required_skills": ["Python", "PyTorch", "Diffusion Models", "Computer Vision", "HuggingFace"],
            "preferred_skills": ["Diffusers", "Gradio", "Docker", "ONNX"],
            "education_requirements": "B.Tech/M.Tech in CS/AI",
            "experience_requirements": "Projects exploring text-to-image or multimodal generative pipelines",
            "location": "Bengaluru, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Knowledge of noise scheduling, classifier-free guidance, and latent spaces.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-GENAI-005",
            "title": "Autonomous AI Agents Intern",
            "company_name": "AgenticCore Labs [Synthetic Benchmark]",
            "category": "Generative AI",
            "description": "Construct multi-agent cooperative workflows, tool-calling pipelines, and recursive task planners using LangGraph and AutoGen.",
            "required_skills": ["Python", "LangChain", "Large Language Models", "Tool Calling", "APIs"],
            "preferred_skills": ["LangGraph", "FastAPI", "Asyncio", "Pydantic"],
            "education_requirements": "B.Tech in Computer Science or Software Engineering",
            "experience_requirements": "Personal or academic projects building autonomous agents with external API access",
            "location": "Gurugram, India",
            "work_mode": "Remote",
            "duration": "4 months",
            "other_requirements": "Understanding of ReAct loops, planning strategies, and state graphs.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-GENAI-006",
            "title": "Generative AI Evaluation & Safety Intern",
            "company_name": "SafeGen Guardrails [Synthetic Benchmark]",
            "category": "Generative AI",
            "description": "Benchmark LLM guardrails, jailbreak prevention, toxicity classifiers, and automated hallucination evaluation metrics (RAGAS, G-Eval).",
            "required_skills": ["Python", "NLP", "LLM Evaluation", "Pandas", "Scikit-Learn"],
            "preferred_skills": ["RAGAS", "NeMo Guardrails", "HuggingFace", "Regex"],
            "education_requirements": "B.Tech/M.Tech in CS, AI, or Information Security",
            "experience_requirements": "Coursework or projects analyzing language model safety and alignment",
            "location": "Bengaluru, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Methodical approach to prompt red-teaming and safety benchmarks.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },

        # 9. Java Development (6 records)
        {
            "internship_id": "INT-JAVA-001",
            "title": "Java Spring Boot Backend Intern",
            "company_name": "Enterprise Logic Works [Synthetic Benchmark]",
            "category": "Java Development",
            "description": "Engineer scalable RESTful microservices using Java 17, Spring Boot, Spring Data JPA, Hibernate, and PostgreSQL.",
            "required_skills": ["Java", "Spring Boot", "Spring Data JPA", "Hibernate", "SQL"],
            "preferred_skills": ["PostgreSQL", "Maven", "Docker", "JUnit"],
            "education_requirements": "B.Tech / B.E. / MCA in Computer Science or Information Technology",
            "experience_requirements": "Hands-on projects building REST controllers, services, and repositories in Spring Boot",
            "location": "Bengaluru, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Strong command of Object-Oriented Programming (SOLID), streams, and collections.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-JAVA-002",
            "title": "Java Microservices & Cloud Intern",
            "company_name": "JavaCloud Microgrid [Synthetic Benchmark]",
            "category": "Java Development",
            "description": "Build distributed microservices with Spring Cloud, Eureka service discovery, API Gateway, and resilient circuit breakers.",
            "required_skills": ["Java", "Spring Boot", "Spring Cloud", "REST APIs", "Microservices"],
            "preferred_skills": ["Resilience4j", "Docker", "Kafka", "Kubernetes"],
            "education_requirements": "B.Tech in Computer Science or Software Engineering",
            "experience_requirements": "Academic projects involving multi-service architectures in Java",
            "location": "Hyderabad, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Knowledge of synchronous vs asynchronous inter-service communication.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-JAVA-003",
            "title": "Java Core & Algorithms Intern",
            "company_name": "Algorithmic Java Systems [Synthetic Benchmark]",
            "category": "Java Development",
            "description": "Optimize core Java backend systems, refactor legacy batch services, and enforce rigorous unit testing with JUnit 5 and Mockito.",
            "required_skills": ["Java", "Data Structures", "Algorithms", "Object-Oriented Programming", "JUnit"],
            "preferred_skills": ["Mockito", "Maven", "Git", "Design Patterns"],
            "education_requirements": "B.E./B.Tech/MCA in Computer Science",
            "experience_requirements": "Strong competitive programming or LeetCode problem solving background in Java",
            "location": "Pune, India",
            "work_mode": "Hybrid",
            "duration": "3 months",
            "other_requirements": "Understanding of JVM memory model, garbage collection, and threading.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-JAVA-004",
            "title": "Java Messaging & Event-Driven Systems Intern",
            "company_name": "StreamPulse Event Labs [Synthetic Benchmark]",
            "category": "Java Development",
            "description": "Implement high-throughput event publishers and consumers using Java, Apache Kafka, and Spring for Kafka.",
            "required_skills": ["Java", "Spring Boot", "Kafka", "Event-Driven Architecture", "SQL"],
            "preferred_skills": ["Avro", "Docker", "PostgreSQL", "Gradle"],
            "education_requirements": "B.Tech in CS/IT",
            "experience_requirements": "Familiarity with pub-sub message brokers and distributed messaging",
            "location": "Chennai, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Knowledge of partition offsets, consumer groups, and idempotency.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-JAVA-005",
            "title": "Java Security & Identity Engineering Intern",
            "company_name": "AuthSecure Java Systems [Synthetic Benchmark]",
            "category": "Java Development",
            "description": "Integrate Spring Security, OAuth2, and JSON Web Tokens (JWT) into existing enterprise Java services.",
            "required_skills": ["Java", "Spring Boot", "Spring Security", "OAuth2", "JWT"],
            "preferred_skills": ["Keycloak", "PostgreSQL", "REST APIs", "JUnit"],
            "education_requirements": "B.Tech/MCA in Computer Science or Cyber Security",
            "experience_requirements": "Coursework or projects configuring role-based access control (RBAC)",
            "location": "Noida, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Familiarity with security filter chains and cryptographic hashing.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-JAVA-006",
            "title": "Java FinTech & Transaction Systems Intern",
            "company_name": "FinPay Java Architectures [Synthetic Benchmark]",
            "category": "Java Development",
            "description": "Develop high-reliability payment transaction processors with distributed locking and strict ACID compliance.",
            "required_skills": ["Java", "Spring Boot", "SQL", "Database Transactions", "REST APIs"],
            "preferred_skills": ["Redis", "Oracle/PostgreSQL", "Docker", "Microservices"],
            "education_requirements": "B.Tech/B.E. in CS/IT",
            "experience_requirements": "College projects handling financial or e-commerce transactions",
            "location": "Mumbai, India",
            "work_mode": "On-site",
            "duration": "6 months",
            "other_requirements": "Emphasis on 99.99% uptime mindset and two-phase commit concepts.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },

        # 10. Data Analytics (6 records)
        {
            "internship_id": "INT-DA-001",
            "title": "Business Intelligence & Data Analytics Intern",
            "company_name": "MetricStream BI Labs [Synthetic Benchmark]",
            "category": "Data Analytics",
            "description": "Transform messy operational data into high-impact interactive dashboards, perform cohort analysis, and write complex SQL aggregations.",
            "required_skills": ["Python", "Data Analysis", "SQL", "Pandas", "PowerBI"],
            "preferred_skills": ["Tableau", "Excel / Advanced DAX", "Matplotlib", "Seaborn"],
            "education_requirements": "B.Tech / B.Sc / BCA / BBA (Analytics) / MCA (2024-2026)",
            "experience_requirements": "Proficiency in window functions, CTEs, and exploratory visual dashboards",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Strong storytelling skills translating quantitative findings to operational stakeholders.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-DA-002",
            "title": "Tableau & Visual Analytics Intern",
            "company_name": "VisualIQ Intelligence [Synthetic Benchmark]",
            "category": "Data Analytics",
            "description": "Construct executive performance scorecards and interactive drill-down reports in Tableau Desktop.",
            "required_skills": ["Tableau", "SQL", "Data Visualization", "Excel", "Data Cleaning"],
            "preferred_skills": ["Python", "Pandas", "Alteryx"],
            "education_requirements": "Degree in Analytics, Information Systems, Business, or Computer Science",
            "experience_requirements": "Tableau Public portfolio showcasing interactive dashboards",
            "location": "Pune, India",
            "work_mode": "Hybrid",
            "duration": "3 months",
            "other_requirements": "Grasp of visual hierarchy, Gestalt principles, and calculation fields.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-DA-003",
            "title": "Marketing Analytics & Growth Intern",
            "company_name": "GrowthMetric Labs [Synthetic Benchmark]",
            "category": "Data Analytics",
            "description": "Analyze multi-channel marketing campaigns, attribution funnels, CAC/LTV metrics, and user journey drop-offs.",
            "required_skills": ["Python", "SQL", "Pandas", "Data Analytics", "Google Analytics"],
            "preferred_skills": ["Excel", "PowerBI", "Funnel Analysis"],
            "education_requirements": "B.Tech/BBA/B.Sc in Economics, Statistics, or Analytics",
            "experience_requirements": "Experience analyzing web analytics or campaign conversion data",
            "location": "Gurugram, India",
            "work_mode": "Remote",
            "duration": "4 months",
            "other_requirements": "Understanding of ROAS, churn rates, and cohort retention decay.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-DA-004",
            "title": "Supply Chain & Operations Analytics Intern",
            "company_name": "LogiOptima Analytics [Synthetic Benchmark]",
            "category": "Data Analytics",
            "description": "Track vendor fulfillment rates, warehouse inventory turns, and transit lead-time deviations using SQL and Python.",
            "required_skills": ["SQL", "Python", "Data Analysis", "Excel", "Operations Research"],
            "preferred_skills": ["PowerBI", "Statsmodels", "Pandas"],
            "education_requirements": "B.Tech in Industrial Engineering, Mechanical, CS, or Supply Chain Management",
            "experience_requirements": "Academic projects dealing with inventory models or logistics tracking",
            "location": "Chennai, India",
            "work_mode": "Hybrid",
            "duration": "6 months",
            "other_requirements": "Familiarity with safety stock formulas and reorder point calculations.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-DA-005",
            "title": "Financial Analytics & Reporting Intern",
            "company_name": "FinView Analytics [Synthetic Benchmark]",
            "category": "Data Analytics",
            "description": "Automate variance analysis across quarterly P&L balance sheets and assemble executive KPI presentations.",
            "required_skills": ["Excel", "SQL", "Data Analytics", "Financial Analysis", "PowerBI"],
            "preferred_skills": ["Python", "Pandas", "VBA"],
            "education_requirements": "Degree in Finance, Commerce, Economics, or Analytics",
            "experience_requirements": "Coursework in corporate finance and advanced financial modeling in Excel",
            "location": "Mumbai, India",
            "work_mode": "On-site",
            "duration": "6 months",
            "other_requirements": "High diligence with numerical reconciliation and zero-error tolerance.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        },
        {
            "internship_id": "INT-DA-006",
            "title": "Product Analytics & User Telemetry Intern",
            "company_name": "PulseProduct Insights [Synthetic Benchmark]",
            "category": "Data Analytics",
            "description": "Analyze in-app user telemetry events, feature adoption curves, and session retention drops using SQL and Amplitude/Mixpanel concepts.",
            "required_skills": ["SQL", "Python", "Data Analytics", "Pandas", "User Behavior Analysis"],
            "preferred_skills": ["Product Analytics", "Tableau", "Seaborn"],
            "education_requirements": "B.Tech/BCA/B.Sc in CS or Data Analytics",
            "experience_requirements": "Experience querying nested JSON event tables and funnel metrics",
            "location": "Bengaluru, India",
            "work_mode": "Remote",
            "duration": "6 months",
            "other_requirements": "Curiosity about onboarding funnels, DAU/MAU ratios, and feature stickiness.",
            "is_synthetic": True,
            "provenance": "Infosys AI/ML Internship Benchmark Dataset v1.0"
        }
    ]
    return internships


def build_test_candidates_dataset() -> List[Dict[str, Any]]:
    """
    Builds the 10 core candidate benchmark profiles (from Module 7)
    plus 10 edge case candidates (from Module 8).
    """
    candidates = [
        # --- 10 MODULE 7 BENCHMARK SCENARIOS ---
        {
            "candidate_id": "CAND-01",
            "candidate_name": "Aarav Sharma",
            "email": "aarav.sharma.test@example.com",
            "target_category": "Backend Development",
            "scenario_description": "Python + Django candidate -> Backend internships",
            "education": {
                "degree": "B.Tech",
                "specialization": "Computer Science and Engineering",
                "college": "National Institute of Technology, Karnataka",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["Python", "SQL"],
                "frameworks": ["Django", "Django REST Framework"],
                "tools": ["Git", "PostgreSQL", "Docker", "Postman", "Linux"]
            },
            "projects": [
                {
                    "title": "E-Commerce Micro-Backend Service",
                    "description": "Architected a high-throughput Django REST service with JWT authentication, PostgreSQL relational schema, and Redis caching for product catalog queries.",
                    "technologies": ["Python", "Django", "Django REST Framework", "PostgreSQL", "Redis"]
                },
                {
                    "title": "Automated Invoice Billing API",
                    "description": "Implemented idempotent billing webhooks and PDF invoice generation backend in Django with Celery asynchronous task workers.",
                    "technologies": ["Python", "Django", "Celery", "PostgreSQL"]
                }
            ],
            "internship_experience": [
                {
                    "role": "Backend Intern",
                    "company": "TechVenture Labs [Academic]",
                    "duration": "3 months",
                    "description": "Built CRUD REST endpoints using Django and DRF; improved SQL query latency by adding appropriate database indexes."
                }
            ],
            "work_experience": [],
            "certifications": ["Meta Back-End Developer Specialization", "PostgreSQL for Everybody"],
            "other_information": "Enthusiastic about scalable backend architectures and API security."
        },
        {
            "candidate_id": "CAND-02",
            "candidate_name": "Priya Patel",
            "email": "priya.patel.test@example.com",
            "target_category": "Machine Learning",
            "scenario_description": "Python + Machine Learning candidate -> AI/ML internships",
            "education": {
                "degree": "B.E.",
                "specialization": "Artificial Intelligence and Data Science",
                "college": "Pune Institute of Computer Technology",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["Python", "R", "SQL"],
                "frameworks": ["Scikit-Learn", "XGBoost", "LightGBM", "PyTorch"],
                "tools": ["Pandas", "NumPy", "Matplotlib", "Seaborn", "JupyterLab", "Git"]
            },
            "projects": [
                {
                    "title": "Customer Churn Prediction Engine",
                    "description": "Developed gradient-boosted decision tree pipeline (XGBoost) achieving 0.89 ROC-AUC with extensive feature engineering and SHAP explainability.",
                    "technologies": ["Python", "Scikit-Learn", "XGBoost", "Pandas", "SHAP"]
                },
                {
                    "title": "Predictive Maintenance for Industrial Sensors",
                    "description": "Trained random forest and isolation forest models on multivariate time-series sensor data to detect anomalous equipment failure.",
                    "technologies": ["Python", "Scikit-Learn", "NumPy", "Matplotlib"]
                }
            ],
            "internship_experience": [],
            "work_experience": [],
            "certifications": ["Machine Learning Specialization by Andrew Ng", "Kaggle Competitions Master (Top 5%)"],
            "other_information": "Passionate about tabular data modeling and production ML pipelines."
        },
        {
            "candidate_id": "CAND-03",
            "candidate_name": "Rohan Deshmukh",
            "email": "rohan.deshmukh.test@example.com",
            "target_category": "Data Science",
            "scenario_description": "SQL + Statistics candidate -> Data Science internships",
            "education": {
                "degree": "M.Sc",
                "specialization": "Applied Statistics and Informatics",
                "college": "Indian Statistical Institute",
                "graduation_year": 2024
            },
            "technical_skills": {
                "programming_languages": ["Python", "R", "SQL"],
                "frameworks": ["SciPy", "Statsmodels", "Scikit-Learn"],
                "tools": ["SQL", "Pandas", "Excel", "Tableau", "Git"]
            },
            "projects": [
                {
                    "title": "A/B Testing Framework & Causal Inference",
                    "description": "Formulated hypothesis tests, sample size estimation, and variance reduction (CUPED) for user conversion experiments across 100k samples.",
                    "technologies": ["Python", "Statistics", "Hypothesis Testing", "SQL", "SciPy"]
                },
                {
                    "title": "Econometric Time Series Modeling",
                    "description": "Built ARIMA and vector autoregression (VAR) models to evaluate macroeconomic inflation indicators and stationarity.",
                    "technologies": ["R", "Python", "Statsmodels", "Pandas"]
                }
            ],
            "internship_experience": [
                {
                    "role": "Data Science Intern",
                    "company": "StatMetrics Advisory",
                    "duration": "4 months",
                    "description": "Cleaned multi-million row SQL datasets, constructed regression models, and verified homoscedasticity assumptions."
                }
            ],
            "work_experience": [],
            "certifications": ["Applied Data Science with Python - Coursera", "Advanced SQL for Data Scientists"],
            "other_information": "Strong foundation in Bayesian inference, probability distributions, and scientific computing."
        },
        {
            "candidate_id": "CAND-04",
            "candidate_name": "Ananya Iyer",
            "email": "ananya.iyer.test@example.com",
            "target_category": "Frontend Development",
            "scenario_description": "React + JavaScript candidate -> Frontend internships",
            "education": {
                "degree": "B.Tech",
                "specialization": "Information Technology",
                "college": "Vellore Institute of Technology",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["JavaScript", "TypeScript", "HTML5", "CSS3"],
                "frameworks": ["React", "Next.js", "Tailwind CSS", "Redux Toolkit"],
                "tools": ["Git", "Webpack", "Vite", "Figma", "VS Code", "Jest"]
            },
            "projects": [
                {
                    "title": "Interactive Analytics SaaS Dashboard",
                    "description": "Engineered a responsive single page application in React 18 and Tailwind CSS featuring dark mode, charts, and optimistic UI updates.",
                    "technologies": ["React", "JavaScript", "Tailwind CSS", "Redux Toolkit", "Chart.js"]
                },
                {
                    "title": "Real-time Collaborative Whiteboard",
                    "description": "Constructed high-performance canvas drawing tool in React with WebSockets and undo/redo state stacks.",
                    "technologies": ["React", "TypeScript", "HTML5 Canvas", "CSS3"]
                }
            ],
            "internship_experience": [
                {
                    "role": "Frontend Web Intern",
                    "company": "PixelCanvas Media",
                    "duration": "2 months",
                    "description": "Converted Figma wireframes into responsive React components; improved Lighthouse performance score from 68 to 94."
                }
            ],
            "work_experience": [],
            "certifications": ["Meta Front-End Developer Certificate", "Responsive Web Design - freeCodeCamp"],
            "other_information": "Active contributor to open-source UI component libraries."
        },
        {
            "candidate_id": "CAND-05",
            "candidate_name": "Aditya Verma",
            "email": "aditya.verma.test@example.com",
            "target_category": "Generative AI",
            "scenario_description": "Python + NLP + LLM candidate -> Generative AI internships",
            "education": {
                "degree": "B.Tech",
                "specialization": "Computer Science with AI Specialization",
                "college": "IIIT Hyderabad",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["Python", "SQL"],
                "frameworks": ["LangChain", "LlamaIndex", "HuggingFace", "PyTorch"],
                "tools": ["ChromaDB", "FastAPI", "Docker", "Git", "OpenAI API", "Sentence Transformers"]
            },
            "projects": [
                {
                    "title": "Domain-Specific Document RAG Search System",
                    "description": "Implemented multi-stage Retrieval-Augmented Generation pipeline using ChromaDB, all-MiniLM-L6-v2 embeddings, and Gemini LLM synthesis.",
                    "technologies": ["Python", "ChromaDB", "Sentence Transformers", "LangChain", "RAG", "LLMs"]
                },
                {
                    "title": "Medical Knowledge Assistant with Fine-tuned LoRA",
                    "description": "Fine-tuned open LLM (Llama 3 8B) with LoRA on instruction dataset for biomedical entity recognition and clinical QA.",
                    "technologies": ["Python", "PyTorch", "HuggingFace", "LoRA", "NLP"]
                }
            ],
            "internship_experience": [
                {
                    "role": "AI Research Intern",
                    "company": "NeuralText Labs",
                    "duration": "4 months",
                    "description": "Benchmarked retrieval chunking strategies, dense vs sparse hybrid search, and hallucination scoring on proprietary PDFs."
                }
            ],
            "work_experience": [],
            "certifications": ["DeepLearning.AI Generative AI for Everyone", "HuggingFace NLP Course Completion"],
            "other_information": "Researcher in retrieval evaluation metrics (Recall@K, MRR) and vector database indexing."
        },
        {
            "candidate_id": "CAND-06",
            "candidate_name": "Siddharth Nair",
            "email": "siddharth.nair.test@example.com",
            "target_category": "Java Development",
            "scenario_description": "Java + Spring Boot candidate -> Java Backend internships",
            "education": {
                "degree": "B.E.",
                "specialization": "Computer Science and Engineering",
                "college": "College of Engineering, Guindy",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["Java", "SQL"],
                "frameworks": ["Spring Boot", "Spring Data JPA", "Hibernate", "Spring Security"],
                "tools": ["Maven", "PostgreSQL", "Docker", "Git", "JUnit", "Mockito", "Postman"]
            },
            "projects": [
                {
                    "title": "Digital Banking Microservices Architecture",
                    "description": "Constructed multi-module Spring Boot backend managing customer accounts, transactions, and JWT authentication with PostgreSQL and Maven.",
                    "technologies": ["Java", "Spring Boot", "Spring Data JPA", "PostgreSQL", "JWT"]
                },
                {
                    "title": "Inventory Management REST APIs",
                    "description": "Engineered transactional order booking system enforcing ACID constraints and automated testing with JUnit 5 and Mockito.",
                    "technologies": ["Java", "Spring Boot", "Hibernate", "JUnit", "Maven"]
                }
            ],
            "internship_experience": [],
            "work_experience": [],
            "certifications": ["Oracle Certified Associate, Java SE Programmer", "Spring Framework Masterclass - Udemy"],
            "other_information": "Well versed in Java OOP, multi-threading, concurrency, and SOLID principles."
        },
        {
            "candidate_id": "CAND-07",
            "candidate_name": "Kavya Menon",
            "email": "kavya.menon.test@example.com",
            "target_category": "Software Development",
            "scenario_description": "C++ + DSA candidate -> Software Development / Core Engineering internships",
            "education": {
                "degree": "B.Tech",
                "specialization": "Computer Science and Engineering",
                "college": "Indian Institute of Technology, Madras",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["C++", "C", "Python"],
                "frameworks": ["STL", "Boost", "Google Test"],
                "tools": ["GDB", "Valgrind", "CMake", "Git", "Linux"]
            },
            "projects": [
                {
                    "title": "High-Performance In-Memory Key-Value Store",
                    "description": "Implemented thread-safe lock-free hash table in C++20 with custom memory allocator and LRU eviction policy.",
                    "technologies": ["C++", "Data Structures", "Algorithms", "Concurrency", "CMake"]
                },
                {
                    "title": "Network Packet Sniffer & Flow Analyzer",
                    "description": "Developed low-latency socket packet capture engine decoding TCP/UDP headers and calculating jitter.",
                    "technologies": ["C++", "Socket Programming", "Linux", "GDB"]
                }
            ],
            "internship_experience": [],
            "work_experience": [],
            "certifications": ["Codeforces Candidate Master (Rating: 1940)", "LeetCode 500+ Problems Solved"],
            "other_information": "Top 1% in algorithmic problem solving and low-level systems programming."
        },
        {
            "candidate_id": "CAND-08",
            "candidate_name": "Vikram Singh",
            "email": "vikram.singh.test@example.com",
            "target_category": "Artificial Intelligence",
            "scenario_description": "TensorFlow + Deep Learning candidate -> Deep Learning / AI internships",
            "education": {
                "degree": "B.Tech",
                "specialization": "Artificial Intelligence and Machine Learning",
                "college": "Delhi Technological University",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["Python", "C++"],
                "frameworks": ["TensorFlow", "Keras", "PyTorch", "OpenCV"],
                "tools": ["NumPy", "Matplotlib", "Albumentations", "Git", "Google Colab", "Docker"]
            },
            "projects": [
                {
                    "title": "Autonomous Vehicle Lane & Obstacle Segmentation",
                    "description": "Trained deep convolutional UNet in TensorFlow/Keras on Cityscapes dataset achieving 0.81 mIoU with real-time inference.",
                    "technologies": ["Python", "TensorFlow", "Keras", "Computer Vision", "OpenCV", "Deep Learning"]
                },
                {
                    "title": "Facial Emotion & Mask Detection",
                    "description": "Fine-tuned MobileNetV2 with transfer learning for edge deployment on Raspberry Pi with OpenCV video stream.",
                    "technologies": ["Python", "TensorFlow", "OpenCV", "Transfer Learning"]
                }
            ],
            "internship_experience": [
                {
                    "role": "Computer Vision Intern",
                    "company": "AutoVision Technologies",
                    "duration": "3 months",
                    "description": "Optimized image augmentation pipelines and converted TensorFlow weights to TFLite for mobile execution."
                }
            ],
            "work_experience": [],
            "certifications": ["Deep Learning Specialization - Coursera", "TensorFlow Developer Certificate"],
            "other_information": "Experienced with GPU training clusters and convolutional neural architectures."
        },
        {
            "candidate_id": "CAND-09",
            "candidate_name": "Neha Kulkarni",
            "email": "neha.kulkarni.test@example.com",
            "target_category": "Data Analytics",
            "scenario_description": "Python + Data Analysis candidate -> Data Analytics internships",
            "education": {
                "degree": "B.Tech",
                "specialization": "Computer Science and Business Systems",
                "college": "Savitribai Phule Pune University",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["Python", "SQL"],
                "frameworks": ["Pandas", "NumPy"],
                "tools": ["PowerBI", "Tableau", "Excel", "Matplotlib", "Seaborn", "Git"]
            },
            "projects": [
                {
                    "title": "Sales Revenue & Cohort Retention Dashboard",
                    "description": "Cleaned 500k row transactional sales dataset in Python Pandas, constructed SQL aggregations, and authored PowerBI executive dashboard.",
                    "technologies": ["Python", "Data Analysis", "SQL", "Pandas", "PowerBI", "Excel"]
                },
                {
                    "title": "Customer Segmentation & RFM Analysis",
                    "description": "Calculated Recency, Frequency, and Monetary scores on retail data to categorize high-value VIP shopper cohorts.",
                    "technologies": ["Python", "Pandas", "Seaborn", "Data Analytics"]
                }
            ],
            "internship_experience": [
                {
                    "role": "Business Analyst Intern",
                    "company": "Kurate Data Insights",
                    "duration": "3 months",
                    "description": "Drafted automated weekly KPI performance sheets in Excel and created SQL data extracts for business leadership."
                }
            ],
            "work_experience": [],
            "certifications": ["Google Data Analytics Professional Certificate", "Microsoft Certified: Power BI Data Analyst Associate"],
            "other_information": "Strong passion for business metric definitions and data-driven storytelling."
        },
        {
            "candidate_id": "CAND-10",
            "candidate_name": "Arjun Singhania",
            "email": "arjun.singhania.test@example.com",
            "target_category": "Full Stack Development",
            "scenario_description": "Python + Django + React candidate -> Full Stack internships",
            "education": {
                "degree": "B.Tech",
                "specialization": "Information Technology",
                "college": "Manipal Institute of Technology",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["Python", "JavaScript", "SQL", "HTML5", "CSS3"],
                "frameworks": ["Django", "Django REST Framework", "React", "Tailwind CSS"],
                "tools": ["PostgreSQL", "Docker", "Git", "Postman", "Linux", "Node.js"]
            },
            "projects": [
                {
                    "title": "Enterprise Project Management Web Suite",
                    "description": "Developed full stack platform with React SPA frontend, Tailwind CSS styles, and Django REST backend backed by PostgreSQL with JWT auth.",
                    "technologies": ["Python", "Django", "React", "JavaScript", "PostgreSQL", "Tailwind CSS"]
                },
                {
                    "title": "Real-time Polls & Discussion Platform",
                    "description": "Implemented Django Channels WebSocket consumers and React client-side live chart visualizations.",
                    "technologies": ["Python", "Django", "React", "WebSockets", "SQL"]
                }
            ],
            "internship_experience": [
                {
                    "role": "Full Stack Intern",
                    "company": "AppCraft Solutions",
                    "duration": "4 months",
                    "description": "Built full-stack features from frontend forms in React to backend API views in Django; containerized application with Docker Compose."
                }
            ],
            "work_experience": [],
            "certifications": ["Full Stack Web Development with React and Python", "Docker Essentials"],
            "other_information": "Enjoys bridging backend API services with clean, reactive user interfaces."
        },

        # --- 10 MODULE 8 ADVANCED EDGE CASES ---
        {
            "candidate_id": "EDGE-01",
            "candidate_name": "Edge NoSkills Candidate",
            "email": "edge.noskills@example.com",
            "target_category": "Edge Case - No Skills",
            "scenario_description": "Resume with no technical skills (empty skills dictionary/lists)",
            "education": {
                "degree": "B.A.",
                "specialization": "English Literature",
                "college": "City College of Arts",
                "graduation_year": 2024
            },
            "technical_skills": {
                "programming_languages": [],
                "frameworks": [],
                "tools": []
            },
            "projects": [],
            "internship_experience": [],
            "work_experience": [],
            "certifications": [],
            "other_information": "Interested in exploring tech opportunities with a non-technical background."
        },
        {
            "candidate_id": "EDGE-02",
            "candidate_name": "Edge MissingEdu Candidate",
            "email": "edge.missingedu@example.com",
            "target_category": "Edge Case - Missing Education",
            "scenario_description": "Resume with missing education information (null/empty values)",
            "education": {
                "degree": None,
                "specialization": None,
                "college": None,
                "graduation_year": None
            },
            "technical_skills": {
                "programming_languages": ["Python", "SQL"],
                "frameworks": ["Django"],
                "tools": ["Git", "PostgreSQL"]
            },
            "projects": [
                {
                    "title": "Self-Taught Django Backend",
                    "description": "Built standalone REST APIs in Django without formal degree credentials.",
                    "technologies": ["Python", "Django", "SQL"]
                }
            ],
            "internship_experience": [],
            "work_experience": [],
            "certifications": ["Self Taught Programmer Certificate"],
            "other_information": "Self-directed developer with no university background listed."
        },
        {
            "candidate_id": "EDGE-03",
            "candidate_name": "Edge Polyglot Candidate",
            "email": "edge.polyglot@example.com",
            "target_category": "Edge Case - Multiple Languages",
            "scenario_description": "Resume with 7+ multiple programming languages and diverse frameworks",
            "education": {
                "degree": "B.Tech",
                "specialization": "Computer Science",
                "college": "State Technical University",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["Python", "Java", "C++", "JavaScript", "Go", "Rust", "Ruby", "PHP", "Kotlin", "Swift"],
                "frameworks": ["Django", "Spring Boot", "React", "Vue.js", "Express", "Flask", "Flutter"],
                "tools": ["Git", "Docker", "Kubernetes", "Linux", "PostgreSQL", "MongoDB", "Redis"]
            },
            "projects": [
                {
                    "title": "Cross-Language Benchmarking Utility",
                    "description": "Built benchmark suite testing concurrency in Go, Rust, Java, and Python.",
                    "technologies": ["Go", "Rust", "Java", "Python"]
                }
            ],
            "internship_experience": [],
            "work_experience": [],
            "certifications": [],
            "other_information": "Polyglot software enthusiast experimenting with many ecosystems."
        },
        {
            "candidate_id": "EDGE-04",
            "candidate_name": "Edge UnrelatedProjects Candidate",
            "email": "edge.unrelated@example.com",
            "target_category": "Edge Case - Unrelated Projects",
            "scenario_description": "Resume with unrelated projects (music production, culinary website, photography blog)",
            "education": {
                "degree": "B.Sc",
                "specialization": "Media & Fine Arts",
                "college": "Creative Arts Academy",
                "graduation_year": 2024
            },
            "technical_skills": {
                "programming_languages": ["Python"],
                "frameworks": [],
                "tools": ["Audacity", "Photoshop", "Lightroom", "WordPress"]
            },
            "projects": [
                {
                    "title": "Audio Mastering & Sound Design Portfolio",
                    "description": "Composed 12 soundtrack tracks and mixed audio levels for indie films.",
                    "technologies": ["Audacity", "Logic Pro", "Sound Editing"]
                },
                {
                    "title": "Artisanal Bakery Recipe Blog",
                    "description": "Authored food photography articles on a hosted WordPress installation.",
                    "technologies": ["WordPress", "Photography", "SEO"]
                }
            ],
            "internship_experience": [],
            "work_experience": [],
            "certifications": ["Sound Design Masterclass"],
            "other_information": "Creative artist looking to see if python scripting skills match any tech role."
        },
        {
            "candidate_id": "EDGE-05",
            "candidate_name": "Edge AcademicOnly Candidate",
            "email": "edge.academiconly@example.com",
            "target_category": "Edge Case - Academic Only No Experience",
            "scenario_description": "Resume with only theoretical academic lab assignments and zero experience/internships",
            "education": {
                "degree": "B.Tech",
                "specialization": "Computer Science",
                "college": "Regional Engineering College",
                "graduation_year": 2026
            },
            "technical_skills": {
                "programming_languages": ["C", "C++", "Java"],
                "frameworks": [],
                "tools": ["Turbo C++", "Eclipse", "Linux Command Line"]
            },
            "projects": [
                {
                    "title": "College Operating Systems Lab Assignment",
                    "description": "Implemented round robin CPU scheduling simulator and semaphore synchronization in C for college coursework lab.",
                    "technologies": ["C", "Operating Systems", "Algorithms"]
                },
                {
                    "title": "DBMS Coursework Hospital Record Script",
                    "description": "Created normalized 3NF relational tables and basic SQL queries for DBMS lab examination.",
                    "technologies": ["SQL", "DBMS", "MySQL"]
                }
            ],
            "internship_experience": [],
            "work_experience": [],
            "certifications": [],
            "other_information": "Pre-final year student with only syllabus lab coursework."
        },
        {
            "candidate_id": "EDGE-06",
            "candidate_name": "Edge EmptyDb Candidate",
            "email": "edge.emptydb@example.com",
            "target_category": "Backend Development",
            "scenario_description": "Standard candidate used to test resilience when searching an empty internship database",
            "education": {
                "degree": "B.Tech",
                "specialization": "Information Technology",
                "college": "Anna University",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["Python"],
                "frameworks": ["Django"],
                "tools": ["Git", "SQL"]
            },
            "projects": [
                {
                    "title": "Django Blog",
                    "description": "Simple article posting website with comments in Django.",
                    "technologies": ["Python", "Django"]
                }
            ],
            "internship_experience": [],
            "work_experience": [],
            "certifications": [],
            "other_information": "Test profile for database state resilience."
        },
        {
            "candidate_id": "EDGE-07",
            "candidate_name": "Edge PartialMeta Candidate",
            "email": "edge.partialmeta@example.com",
            "target_category": "Machine Learning",
            "scenario_description": "Candidate tested against database containing incomplete or sparse metadata fields",
            "education": {
                "degree": "B.Tech",
                "specialization": "Computer Science",
                "college": "JNTU",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["Python"],
                "frameworks": ["Scikit-Learn"],
                "tools": ["Pandas"]
            },
            "projects": [
                {
                    "title": "Iris Species Classifier",
                    "description": "Trained logistic regression on Iris dataset.",
                    "technologies": ["Python", "Scikit-Learn"]
                }
            ],
            "internship_experience": [],
            "work_experience": [],
            "certifications": [],
            "other_information": "Test candidate for partial metadata handling."
        },
        {
            "candidate_id": "EDGE-08",
            "candidate_name": "Edge DuplicateTest Candidate",
            "email": "edge.duplicatetest@example.com",
            "target_category": "Frontend Development",
            "scenario_description": "Candidate used to verify deduplication behavior when duplicate internships are ingested",
            "education": {
                "degree": "BCA",
                "specialization": "Computer Applications",
                "college": "Bangalore University",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["JavaScript", "HTML5", "CSS3"],
                "frameworks": ["React"],
                "tools": ["Git"]
            },
            "projects": [
                {
                    "title": "Todo List in React",
                    "description": "Simple task list application with local storage.",
                    "technologies": ["React", "JavaScript"]
                }
            ],
            "internship_experience": [],
            "work_experience": [],
            "certifications": [],
            "other_information": "Testing ingestion deduplication stability."
        },
        {
            "candidate_id": "EDGE-09",
            "candidate_name": "Edge NoMatch Niche Candidate",
            "email": "edge.nomatch@example.com",
            "target_category": "Niche - No Available Matches",
            "scenario_description": "Candidate whose specialized skills (Quantum Computing, Qiskit, FPGA, Verilog) do not match standard internships",
            "education": {
                "degree": "B.Tech",
                "specialization": "Electrical Engineering & Quantum Computing",
                "college": "IIT Bombay",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["Verilog", "VHDL", "QASM", "MATLAB"],
                "frameworks": ["Qiskit", "Cirq", "Xilinx Vivado"],
                "tools": ["FPGA", "Oscilloscope", "Logic Analyzer"]
            },
            "projects": [
                {
                    "title": "FPGA Cryptographic Accelerator in Verilog",
                    "description": "Synthesized hardware accelerated SHA-256 pipeline on Xilinx Artix-7 FPGA chip.",
                    "technologies": ["Verilog", "FPGA", "Xilinx Vivado"]
                },
                {
                    "title": "Variational Quantum Eigensolver Simulation",
                    "description": "Simulated ground state molecular energies on noisy quantum circuits using Qiskit.",
                    "technologies": ["Qiskit", "Quantum Computing", "Python"]
                }
            ],
            "internship_experience": [],
            "work_experience": [],
            "certifications": ["IBM Certified Associate Developer - Quantum Computation using Qiskit v0.2X"],
            "other_information": "Hardware and quantum computing specialist looking for standard IT internships."
        },
        {
            "candidate_id": "EDGE-10",
            "candidate_name": "Edge Minimal Candidate",
            "email": "edge.minimal@example.com",
            "target_category": "Minimal / Ultra Short",
            "scenario_description": "Very short resume with 1 line of skills and minimal words",
            "education": {
                "degree": "B.Tech",
                "specialization": "CS",
                "college": "XYZ College",
                "graduation_year": 2025
            },
            "technical_skills": {
                "programming_languages": ["Python"],
                "frameworks": [],
                "tools": []
            },
            "projects": [],
            "internship_experience": [],
            "work_experience": [],
            "certifications": [],
            "other_information": "Python beginner."
        }
    ]
    return candidates


def generate_all_data(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "resumes"), exist_ok=True)

    # 1. Generate internships.json
    internships = build_internships_dataset()
    internships_path = os.path.join(output_dir, "internships.json")
    with open(internships_path, "w", encoding="utf-8") as f:
        json.dump(internships, f, indent=2)
    print(f"Generated {len(internships)} internship postings at: {internships_path}")

    # 2. Generate test_candidates.json
    candidates = build_test_candidates_dataset()
    candidates_path = os.path.join(output_dir, "test_candidates.json")
    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)
    print(f"Generated {len(candidates)} test candidates at: {candidates_path}")

    # 3. Generate individual resume JSON files in data/resumes/
    for cand in candidates:
        fname = f"{cand['candidate_id'].lower()}_{cand['candidate_name'].lower().replace(' ', '_')}.json"
        fpath = os.path.join(output_dir, "resumes", fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(cand, f, indent=2)
    print(f"Generated {len(candidates)} individual resume JSON files in: {os.path.join(output_dir, 'resumes')}")


if __name__ == "__main__":
    generate_all_data()
