# AIML-01 – Real-Time Financial Fraud Detection Platform

Source: Nexvia Technologies student portal (`/student/projects`), Project #2 of 2. Captured 2026-11-26.

- Track: AI / Machine Learning. Batch start 2026-10-29. **Deadline 2026-12-15.** Estimated effort 14 days.
- Goal: read live transactions, score each one for fraud risk in milliseconds, link suspicious
  accounts into fraud rings, and give analysts a dashboard to review cases.
- Tools named: Kafka, XGBoost, PyTorch Geometric, Feast, MLflow, FastAPI, Kubernetes, Evidently AI.

## Milestones from the portal
1. Get a dataset (IEEE-CIS Fraud Detection or PaySim). EDA; note the heavy class imbalance (<1% fraud).
2. Feature engineering: transaction velocity, amount vs the user's average, device and location
   changes. Store them in Feast so training and serving use identical features.
3. XGBoost with class weights or SMOTE. Evaluate with PR-AUC, not accuracy. Track all runs in MLflow.
4. Graph of accounts, devices and cards; a PyTorch Geometric GNN to find connected fraud rings.
   Blend its score with XGBoost.
5. Kafka producer replaying transactions; FastAPI consumer service that fetches features and
   returns a fraud score.
6. Evidently AI data-drift reports and an automatic retraining trigger.
7. Analyst dashboard (React or Streamlit) with alerts, case status and feedback labels. Deploy the
   services on Kubernetes (Minikube is fine).

## Dataset options
IEEE-CIS Fraud Detection; PaySim; Credit Card Fraud (ULB, 284,807 tx); Bank Account Fraud (BAF,
NeurIPS 2022); Elliptic Bitcoin graph (for the GNN part). All on Kaggle.

## Deliverables (shared with AIML-04)
- Working demo (deployed or screen-recorded)
- GitHub repository with clean commits, README and setup instructions
- Architecture diagram and data or API documentation
- Test results and performance/accuracy numbers
- 10-minute presentation: problem, design, challenges, results, next steps
- Individual reflection: what I learned and what I owned

## Portal submission rules
- One public GitHub repo URL for both projects, organised as subfolders or branches
  (`AIML-04`, `AIML-01`) with source code, tests and screenshots. This project lives on branch `AIML-01`.
- README with architecture, install/run steps and feature breakdown.
- Optional: Live Deployment URL (Vercel / Netlify / Render), Google Drive URL, reviewer notes.
- **Single submission only**; locked once submitted.
