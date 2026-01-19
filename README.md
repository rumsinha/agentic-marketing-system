Agentic Marketing Intelligence Platform
A hybrid AI system that combines Strategic Rules (RFM) with Machine Learning (HDBSCAN) to provide autonomous marketing insights. The system uses Google ADK (Agent Development Kit) and Gemini to reason about data stored in Snowflake and PostgreSQL.
1. Prerequisites & Setup
Before running the pipeline, ensure you have the following installed and configured:
•	Python 3.10+
•	PostgreSQL (for raw operational data)
•	Snowflake (for analytical queries)
•	Google Gemini API Key
Environment Variables (.env) Create a .env file in the root directory with the following:
# Database Configs
DATABASE_URL=postgresql://user:pass@localhost:5432/olist
SNOWFLAKE_ACCOUNT=...
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...

# AI Configs
GOOGLE_API_KEY=...

2. The Data Pipeline (ETL & ML)
Run these scripts in order to build the intelligence layer.
Step 1: Ingest Raw Data
Loads the Olist E-commerce dataset (CSVs) into PostgreSQL.
•	Script: load_olist_data.py / schema.sql
•	Action: Creates the olist schema and loads customers, orders, and products.
•	Command: python load_olist_data.py --data-dir ./data
Step 2: Data Exploration (Optional)
Generates profiling reports to understand data distribution.
•	Script: data_exploration.py
•	Action: Outputs charts and markdown reports to ./exploration_reports.
•	Command: python data_exploration.py --db-url $DATABASE_URL
Step 3: Feature Engineering
Transforms raw SQL data into machine learning features (RFM, Behavioral, Temporal).
•	Script: feature_engineering.py
•	Action: Creates customer_features.csv and scales data for clustering.
•	Command: python feature_engineering.py --db-url $DATABASE_URL
Step 4: Segmentation & Modeling
The core logic engine. Runs HDBSCAN clustering and assigns RFM segments.
•	Script: hdbscan_clustering.py
•	Action:
o	Assigns every customer an RFM Segment (Strategy).
o	Assigns every customer a Behavioral Cluster (Tactic).
o	Generates marketing_playbook.json (Used by the Agent).
o	Exports customer_cohorts.csv, cohort_profiles.csv, and cluster_profiles.csv.
•	Command: python hdbscan_clustering.py
Step 5: Load to Snowflake
Moves the processed insights into the Data Warehouse for the Agent to query.
•	Script: load data.sql
•	Action: Loads the CSVs generated in Step 4 into Snowflake tables: customer_intelligence, cohort_profiles, and cluster_profiles.
•	Instruction: Run the SQL commands in your Snowflake console or via CLI.
3. Running the Agent
Once the data pipeline is complete and the marketing_playbook.json exists, you can launch the agent.
The Agent Architecture
•	Framework: Google ADK (Agent Development Kit).
•	Brain: Google Gemini 2.5 Flash.
•	Tools: Defined in cohort_tools.py.
•	Memory: Uses InMemorySessionService to remember context (e.g., "Compare that segment...").
Execution Modes
1. Automated Demo Runs a pre-defined script of queries to showcase routing and memory.
python MarketingAgent.py –demo
2. Interactive Chat Chat with your data manually.
python MarketingAgent.py
4. How the Agent Works
The agent uses a Hybrid Intelligence approach to answer questions:
1.	Intent Detection (cohort_tools.py):
o	Analyzes your prompt to decide if you need Strategy (RFM) or Targeting (Clusters).
o	Example: "Who are my best customers?" -> Routes to RFM.
o	Example: "Who buys electronics?" -> Routes to HDBSCAN.
2.	Tool Execution:
o	smart_customer_query: The main router.
o	get_segment_details: Reads from the local marketing_playbook.json.
o	find_behavioral_cluster: Queries Snowflake for granular targeting.
o	compare_segments: Compares two cohorts side-by-side.
3.	Contextual Memory:
o	The agent remembers the last segment discussed.
o	Flow: User asks about "Champions" -> Agent answers -> User says "Compare it with Lost" -> Agent knows "it" means Champions.



<img width="468" height="636" alt="image" src="https://github.com/user-attachments/assets/e04c3013-392a-4a42-90c5-f1ed7386fbca" />
