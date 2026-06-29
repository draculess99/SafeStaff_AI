# SafeStaff AI: ER Wait-Time Risk and Nurse Staffing Decision Support

SafeStaff AI is a polished decision-support prototype built to optimize Emergency Room (ER) staffing and minimize patient wait-time risks. It integrates predictive Machine Learning with multi-agent scheduling optimization under strict human-in-the-loop governance.

---

### ⚠️ Prototype Safety Notice

This project is a decision-support prototype. It uses synthetic Kaggle ER wait-time data and prototype research-module tables unless real hospital data is supplied. It is not clinically validated and should not be used for real patient staffing decisions without hospital-specific validation, governance, monitoring, and human supervisor approval.

---

## 📖 Capstone Story

SafeStaff AI is not only a wait-time prediction model. It is an agentic decision-support workflow. XGBoost predicts ER wait-time pressure. Research modules add operational context such as ESI acuity, arrival surge, bed boarding, fast-track bottlenecks, and fatigue risk. These signals are combined into an adjusted operational risk score and passed into a dynamic AI committee. The committee activates specialist agents, explains the decision, recommends interventions, and routes high-risk cases to human supervisor approval.

---

## 📋 Problem Statement
Emergency Departments frequently struggle with unpredictable patient influxes, leading to spikes in wait times, staff fatigue, clinical errors, and budget overruns. Traditional scheduling systems are static and fail to account for weather surges, viral outbreak surges, or nurse certifications. Supervisors need proactive, data-driven tools that balance patient safety, labor compliance, and operational costs.

---

## 🛠️ Architecture Summary

XGBoost wait-time prediction
↓
Research-module operational signals
↓
Adjusted operational risk
↓
Dynamic committee agent activation
↓
Committee debate summary
↓
Intervention planner
↓
Human supervisor approval / override
↓
Audit log

---

## 🗃️ Dataset Description
The model is trained on a synthetic ER patient flow dataset including:
* `facility_size_beds`: Number of hospital beds (50–400)
* `month` & `day_of_week`: Temporal factors capturing seasonal and weekend influxes
* `visithour`: Visit arrival hour
* `urgency_level`: Average patient acuity level (1 = High, 5 = Low)
* `nurse_to_patient_ratio`: Staffing density metric
* `specialist_availability`: Toggle representing whether on-call specialists are present

---

## 📊 Model Performance Metrics
The predictive backend uses an optimized XGBoost Regressor (`backend/xgboost_model.pkl`):
* **R² Score**: `0.910` (explains 91% of wait-time variance)
* **Mean Absolute Error (MAE)**: `10.50` minutes
* **Root Mean Squared Error (RMSE)**: `13.30` minutes
* **Dataset Size**: 5,000 total rows (4,000 train, 1,000 test)

---

## 🪙 Token-Saving Design
SafeStaff AI implements a low-token local architecture:
* **Deterministic Fallback**: Agent debates, feature importances, and compliance checks are computed locally by default using structured rules.
* **Smart Session Caching**: Prevents redundant processing or LLM calls by invalidating the debate cache only when scenario inputs change.
* **Cumulative Display**: Tracks prompt/response tokens and costs to audit LLM usage transparently.

---

## 🚀 How to Run

```bash
python backend/test_research_modules.py
python backend/smoke_test_app.py
python app.py
```

---

## 📺 Demo Walkthrough

1. Start the app with `python app.py`
2. Enter or select an ER scenario
3. View XGBoost predicted wait time
4. View base staffing risk
5. View research-module signals
6. View adjusted operational risk
7. Open Dynamic Committee Agent Activation
8. Review Committee Debate Summary
9. Review Why the Recommendation Changed
10. Approve or override in Human Supervisor Review
11. Check audit log"# SafeStaff_AI" 
