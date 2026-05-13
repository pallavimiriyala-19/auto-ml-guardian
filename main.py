# auto_ml_guardian/guardian.py

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import ks_2samp # Kolmogorov-Smirnov test for drift
from loguru import logger
import time
from tenacity import retry, stop_after_attempt, wait_fixed

class MonitorAgent:
    """
    Monitors incoming data and model performance for drift and degradation.
    """
    def __init__(self, baseline_data: np.ndarray, baseline_predictions: np.ndarray,
                 metric_threshold: float = 0.8, data_drift_threshold: float = 0.05,
                 performance_metric='f1_score'):
        self.baseline_data = pd.DataFrame(baseline_data)
        self.baseline_predictions = baseline_predictions
        self.metric_threshold = metric_threshold
        self.data_drift_threshold = data_drift_threshold
        self.performance_metric = performance_metric
        logger.info("MonitorAgent initialized. Ready to observe.")

    def _check_data_drift(self, current_data: pd.DataFrame) -> dict:
        drift_status = {"drift_detected": False, "drifting_features": []}
        for col in self.baseline_data.columns:
            # Using KS-test for numerical features
            if pd.api.types.is_numeric_dtype(self.baseline_data[col]) and \
               pd.api.types.is_numeric_dtype(current_data[col]):
                statistic, p_value = ks_2samp(self.baseline_data[col], current_data[col])
                if p_value < self.data_drift_threshold: # A low p-value suggests distributions are different
                    drift_status["drift_detected"] = True
                    drift_status["drifting_features"].append(col)
                    logger.warning(f"Data drift detected in feature '{col}' (p={p_value:.4f})")
            # For categorical, a more complex test like chi-squared might be needed
            # For simplicity, we'll focus on numerical drift for this example
        return drift_status

    def _check_performance_degradation(self, actual_labels: np.ndarray, predictions: np.ndarray) -> dict:
        if self.performance_metric == 'f1_score':
            current_metric = f1_score(actual_labels, predictions, average='weighted')
        elif self.performance_metric == 'accuracy_score':
            current_metric = accuracy_score(actual_labels, predictions)
        else:
            raise ValueError(f"Unsupported metric: {self.performance_metric}")

        # For simplicity, we compare against a fixed threshold.
        # In a real system, it would compare against historical baseline performance.
        degradation_detected = current_metric < self.metric_threshold
        if degradation_detected:
            logger.error(f"Performance degradation detected! Current {self.performance_metric}: {current_metric:.4f}, Threshold: {self.metric_threshold:.4f}")
        else:
            logger.info(f"Performance stable. Current {self.performance_metric}: {current_metric:.4f}")
        return {"degradation_detected": degradation_detected, "current_metric": current_metric}

    def analyze(self, current_data: np.ndarray, current_predictions: np.ndarray, actual_labels: np.ndarray) -> dict:
        current_data_df = pd.DataFrame(current_data, columns=self.baseline_data.columns)
        data_drift_report = self._check_data_drift(current_data_df)
        performance_report = self._check_performance_degradation(actual_labels, current_predictions)

        issue_detected = data_drift_report["drift_detected"] or performance_report["degradation_detected"]
        return {
            "issue_detected": issue_detected,
            "data_drift": data_drift_report,
            "performance_degradation": performance_report,
            "current_data": current_data,
            "current_labels": actual_labels # Pass labels for potential re-training
        }

class AnalyzerAgent:
    """
    Diagnoses the root cause of detected issues.
    """
    def __init__(self):
        logger.info("AnalyzerAgent initialized. Ready to diagnose.")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def diagnose(self, monitor_report: dict) -> dict:
        if not monitor_report["issue_detected"]:
            logger.info("Analyzer: No issues to diagnose. All good!")
            return {"diagnosis": "No issues detected."}

        diagnosis = {"root_cause": "Unknown", "recommendations": []}

        if monitor_report["data_drift"]["drift_detected"]:
            drifting_features = monitor_report["data_drift"]["drifting_features"]
            diagnosis["root_cause"] = "Data drift"
            diagnosis["recommendations"].append(f"Investigate features: {', '.join(drifting_features)}")
            diagnosis["recommendations"].append("Consider collecting more recent data representative of current distributions.")

        if monitor_report["performance_degradation"]["degradation_detected"]:
            current_metric = monitor_report["performance_degradation"]["current_metric"]
            threshold = monitor_report["performance_metric_threshold"] # Assuming this is passed or accessible
            diagnosis["root_cause"] = "Performance degradation"
            diagnosis["recommendations"].append(f"Model performance dropped to {current_metric:.4f}, below threshold.")
            if not monitor_report["data_drift"]["drift_detected"]:
                diagnosis["recommendations"].append("Could be concept drift, model staleness, or hyperparameter issues.")

        logger.warning(f"Analyzer: Diagnosis complete. Root cause: {diagnosis['root_cause']}")
        return {"diagnosis_report": diagnosis, "monitor_report": monitor_report}

class StrategistAgent:
    """
    Proposes actions based on the diagnosis.
    """
    def __init__(self):
        logger.info("StrategistAgent initialized. Ready to strategize.")

    def formulate_plan(self, diagnosis_report: dict) -> dict:
        plan = {"action": "none", "details": "No action needed."}
        if diagnosis_report["diagnosis_report"]["root_cause"] != "No issues detected.":
            plan["action"] = "retrain_and_optimize"
            plan["details"] = "Proposing model re-training with new data and potential hyperparameter optimization."
            logger.info(f"Strategist: Formulated plan: {plan['details']}")
        return {"plan": plan, "diagnosis_report": diagnosis_report}

class ExecutorAgent:
    """
    Executes the proposed plan, interacting with the ML pipeline.
    """
    def __init__(self, model_retrain_fn):
        self.model_retrain_fn = model_retrain_fn
        logger.info("ExecutorAgent initialized. Ready to execute.")

    def execute_plan(self, plan_report: dict, new_data: np.ndarray, new_labels: np.ndarray, human_approved: bool = True) -> dict:
        plan = plan_report["plan"]
        if not human_approved:
            logger.warning("Executor: Plan not human approved. Aborting execution.")
            return {"status": "aborted", "reason": "Human intervention denied."}

        if plan["action"] == "retrain_and_optimize":
            logger.info("Executor: Initiating re-training...")
            try:
                self.model_retrain_fn(new_data, new_labels)
                logger.success("Executor: Re-training completed successfully!")
                return {"status": "success", "action_taken": "retrained_model"}
            except Exception as e:
                logger.error(f"Executor: Failed to re-train model: {e}")
                return {"status": "failed", "reason": str(e)}
        elif plan["action"] == "none":
            logger.info("Executor: No specific action required based on the plan.")
            return {"status": "no_action", "reason": plan["details"]}
        else:
            logger.warning(f"Executor: Unknown action type: {plan['action']}. No action taken.")
            return {"status": "failed", "reason": "Unknown action type."}

class AutoMLGuardian:
    """
    Orchestrates the autonomous ML pipeline monitoring and adaptation.
    """
    def __init__(self, model_predict_fn, model_retrain_fn, baseline_data, baseline_predictions,
                 metric_threshold: float = 0.85, data_drift_threshold: float = 0.05,
                 performance_metric='f1_score', enable_human_in_loop: bool = False):
        self.monitor_agent = MonitorAgent(baseline_data, baseline_predictions,
                                          metric_threshold, data_drift_threshold, performance_metric)
        self.analyzer_agent = AnalyzerAgent()
        self.strategist_agent = StrategistAgent()
        self.executor_agent = ExecutorAgent(model_retrain_fn)
        self.model_predict_fn = model_predict_fn
        self.enable_human_in_loop = enable_human_in_loop
        logger.info("AutoMLGuardian initialized. Ready to protect your models!")

    def monitor(self, new_data: np.ndarray, actual_labels: np.ndarray):
        logger.info("Guardian: Monitoring new batch of data...")
        current_predictions = self.model_predict_fn(new_data)
        
        monitor_report = self.monitor_agent.analyze(new_data, current_predictions, actual_labels)
        monitor_report["performance_metric_threshold"] = self.monitor_agent.metric_threshold # Pass for analyzer

        if monitor_report["issue_detected"]:
            logger.warning("Guardian: Issue detected! Handing over to Analyzer Agent...")
            diagnosis_report = self.analyzer_agent.diagnose(monitor_report)
            plan_report = self.strategist_agent.formulate_plan(diagnosis_report)

            human_approved = True
            if self.enable_human_in_loop and plan_report["plan"]["action"] != "none":
                logger.info("Guardian: Human-in-the-Loop activated. Reviewing proposed action:")
                logger.info(f"Proposed Plan: {plan_report['plan']['details']}")
                user_input = input("Do you approve this action? (yes/no): ").lower()
                human_approved = (user_input == 'yes')
                if not human_approved:
                    logger.warning("Guardian: Action denied by human.")
            
            execution_result = self.executor_agent.execute_plan(
                plan_report,
                monitor_report["current_data"], # Use the monitored data for retraining
                monitor_report["current_labels"],
                human_approved=human_approved
            )
            logger.info(f"Guardian: Execution result: {execution_result['status']}")
        else:
            logger.info("Guardian: No issues detected. Models are performing well.")

# Example of how to structure the package:
# auto_ml_guardian/
# ├── __init__.py
# ├── guardian.py
# ├── agents/
# │   ├── __init__.py
# │   ├── monitor.py (MonitorAgent)
# │   ├── analyzer.py (AnalyzerAgent)
# │   ├── strategist.py (StrategistAgent)
# │   └── executor.py (ExecutorAgent)
# └── core/
#     ├── __init__.py
#     └── utils.py
#
# For this example, all agent classes are kept in guardian.py for simplicity.
