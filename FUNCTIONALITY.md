# FUNCTIONALITY.md: AutoML Guardian Architecture and Data Flow

`auto-ml-guardian` is designed as an intelligent, autonomous system to manage and optimize machine learning models in production environments. It leverages a multi-agent architecture to ensure proactive maintenance, detect issues like data or concept drift, and adaptively respond to preserve model performance.

## 1. Core Architectural Design

The system is built around a distinct, specialized set of AI agents, each responsible for a specific phase of the monitoring, analysis, and remediation pipeline. This modular design promotes scalability, fault tolerance, and clear separation of concerns.

### Agent Breakdown:

*   **Monitor Agent (The Sentinel)**:
    *   **Role**: The primary data ingestion and anomaly detection component.
    *   **Functionality**: Continuously collects and processes real-time production data and corresponding model predictions. It compares these streams against established baselines (e.g., training data distribution, initial model performance metrics).
    *   **Detection Mechanisms**:
        *   **Data Drift**: Utilizes statistical tests (e.g., Kolmogorov-Smirnov test for numerical features, Chi-squared for categorical) to identify shifts in the distribution of input features over time compared to the baseline.
        *   **Performance Degradation**: Calculates key performance metrics (e.g., F1-score, accuracy, precision, recall) on incoming data (assuming ground truth labels become available) and compares them against a predefined threshold or historical performance.
        *   **Concept Drift (Implicitly)**: While not directly measuring concept drift, a drop in model performance without significant data drift often implies concept drift (change in the relationship between features and target).
    *   **Output**: A detailed report indicating whether any issues (data drift, performance degradation) are detected, along with specific features or metrics affected.

*   **Analyzer Agent (The Diagnostician)**:
    *   **Role**: To delve deeper into detected issues and pinpoint their root causes.
    *   **Functionality**: Upon receiving an anomaly report from the Monitor Agent, the Analyzer performs diagnostic investigations.
    *   **Analysis Mechanisms**:
        *   **Feature Importance Analysis**: Can re-evaluate feature importance on recent data to see if feature relevance has shifted.
        *   **Error Profiling**: Analyzes where the model is failing (e.g., specific classes, data segments).
        *   **Correlation Analysis**: Checks for new correlations or changes in existing ones between features or between features and the target.
    *   **Output**: A diagnosis report identifying the likely root cause(s) of the issue (e.g., "significant data drift in feature X," "model underperforming on class Y," "potential concept drift").

*   **Strategist Agent (The Planner)**:
    *   **Role**: To formulate an optimal remediation plan based on the Analyzer's diagnosis.
    *   **Functionality**: Considers the diagnosed root cause, the severity of the issue, and configurable operational policies to propose a concrete action.
    *   **Strategy Mechanisms**:
        *   **Re-training**: If data or concept drift is severe, suggests re-training the model with a fresh, representative dataset.
        *   **Hyperparameter Optimization**: If performance is degrading without clear data drift, might suggest tuning hyperparameters.
        *   **Feature Engineering Review**: Could recommend a review of existing features or the creation of new ones.
        *   **Model Architecture Change**: For persistent issues, might suggest exploring alternative model architectures.
    *   **Output**: A detailed action plan, including the type of intervention and any specific parameters (e.g., "re-train using last 30 days of data," "run Optuna for hyperparameter search").

*   **Executor Agent (The Operator)**:
    *   **Role**: To implement the Strategist's approved plan by interacting with the underlying ML infrastructure.
    *   **Functionality**: Acts as the interface between the Guardian system and the actual ML pipeline.
    *   **Execution Mechanisms**:
        *   **Trigger Re-training Jobs**: Initiates re-training workflows (e.g., calling an MLOps platform API, running a specific Python script).
        *   **Update Model Configurations**: Adjusts model serving configurations, deploys new model versions.
        *   **Rollback**: Potentially initiates a rollback to a previous stable model version if a new deployment fails or causes further degradation.
    *   **Human-in-the-Loop (HITL) Integration**: Before executing high-impact actions, the Executor can pause and request explicit human approval, providing the proposed plan and the rationale from the Analyzer and Strategist agents.

## 2. Data Flow and Interaction

The agents interact in a sequential, event-driven manner, ensuring a logical progression from detection to resolution.

1.  **Production Data Ingestion**:
    *   Raw production data streams continuously into the `Monitor Agent`. This data includes input features for model inference and, crucially, ground truth labels (which may arrive with a delay) for performance evaluation.
    *   Model predictions generated from the production data are also fed into the `Monitor Agent`.

2.  **Monitoring and Alerting**:
    *   The `Monitor Agent` processes the incoming data and predictions.
    *   It performs real-time or batch comparisons against stored baseline data and performance metrics.
    *   If any data drift or performance degradation is detected beyond configured thresholds, the `Monitor Agent` generates an "Issue Detected" event/report.

3.  **Diagnosis Request**:
    *   The "Issue Detected" event triggers the `Analyzer Agent`.
    *   The `Monitor Agent` passes the relevant current data, baseline data, and its preliminary findings to the `Analyzer Agent`.

4.  **Root Cause Analysis**:
    *   The `Analyzer Agent` performs its diagnostic routines, correlating observed issues with potential root causes.
    *   It generates a "Diagnosis Report" detailing the likely cause(s) and any initial recommendations.

5.  **Strategy Formulation**:
    *   The `Diagnosis Report` is passed to the `Strategist Agent`.
    *   The `Strategist Agent` evaluates the diagnosis and formulates an "Action Plan" (e.g., "re-train model," "tune hyperparameters").

6.  **Human-in-the-Loop (Optional Approval)**:
    *   If `human_in_loop` is enabled and the proposed action is deemed critical, the `AutoMLGuardian` orchestrator presents the "Action Plan" to a human operator for review and approval.
    *   The operator receives the diagnosis, proposed strategy, and can approve or deny the execution.

7.  **Execution and Remediation**:
    *   Upon approval (either automatic or human), the "Action Plan" (along with the new data for re-training) is sent to the `Executor Agent`.
    *   The `Executor Agent` interacts with the underlying ML infrastructure to carry out the plan (e.g., triggering a re-training pipeline, deploying a new model version).

8.  **Feedback Loop**:
    *   Once an action is executed, the newly trained/tuned model is deployed.
    *   The new model's predictions and the ongoing production data continue to feed back into the `Monitor Agent`, closing the loop and allowing the system to continuously adapt and improve.

## 3. Design Decisions

*   **Agent-Based Modularity**: Each agent is a self-contained unit with a clear responsibility. This enhances maintainability, testability, and allows for independent development and scaling of each component.
*   **Event-Driven Communication**: Agents communicate via structured reports/events. This decoupling makes the system flexible and extensible, allowing new agents or alternative implementations to be easily plugged in.
*   **Pluggable ML Operations**: The `Executor Agent` is designed with an abstract interface for triggering ML pipeline actions (e.g., `model_retrain_fn`). This allows `auto-ml-guardian` to integrate with various MLOps platforms (e.g., MLflow, Kubeflow, custom scripts) without internal architectural changes.
*   **Configurable Thresholds**: All monitoring thresholds (performance degradation, data drift significance) are configurable, enabling users to tailor the Guardian's sensitivity to their specific use case and business requirements.
*   **Human-in-the-Loop as a Safety Net**: While autonomous, the optional HITL mechanism provides a crucial safety net for critical operations, maintaining human oversight and trust in the automated system.
*   **Robustness with Retry Mechanisms**: Using libraries like `tenacity` for agent operations (e.g., in `AnalyzerAgent`) ensures transient failures don't halt the entire pipeline, promoting system resilience.
*   **Simplified Data Drift Detection**: For this initial version, data drift focuses on statistical tests for numerical features. Future enhancements could include more sophisticated methods for categorical, mixed-type, and high-dimensional data, as well as concept drift specific detection.
