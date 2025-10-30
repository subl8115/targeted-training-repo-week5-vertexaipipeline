"""
Hands-On Coding Exercise: Convert train.py to Vertex AI Pipeline
===============================================================

OBJECTIVE: Complete the missing component by converting the 
old_model.py to a Vertex AI pipeline component.

INSTRUCTIONS:
1. Review the original train.py file
2. Study the train_to_vertex_ai_conversion.py file for patterns
3. Complete the missing component below
4. Test your understanding of function-to-component conversion

"""

from kfp import dsl, components
from kfp.dsl import (
    component,
    pipeline,
    Input,
    Output,
    Model,
    Metrics
)
from google_cloud_pipeline_components.types import artifact_types

# Pipeline Configuration
PIPELINE_NAME = "diabetes-classification-exercise-pipeline"
BASE_IMAGE = "python:3.9"

# Pre-built BigQuery component (already completed for you)
bigquery_query_job_op = components.load_component_from_url(
    'https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/'
    'bigquery-query-job/sha256:'
    'd1cae80bc0de4e5b95b994739c8d0d7d42ce5a4cb17d3c9512eaed14540f6343'
)

# Evaluation component (already completed for you)
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "google-cloud-bigquery",
        "scikit-learn",
        "joblib", 
        "pandas"
    ]
)
def evaluate_model_op(
    test_data: Input[artifact_types.BQTable],
    model: Input[Model],
    metrics: Output[Metrics],
    min_accuracy: float,
    project_id: str,
    bq_location: str
) -> float:
    import re, logging, joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)
    
    uri = test_data.uri
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BQ table from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    test_df = bq_client.query(query).to_dataframe()

    model_obj = joblib.load(model.path)
    
    FEATURE_COLUMNS = ["Pregnancies","PlasmaGlucose","DiastolicBloodPressure",
                       "TricepsThickness","SerumInsulin","BMI","DiabetesPedigree","Age"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Diabetic"]
    
    preds = model_obj.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("test_samples", len(test_df))
    
    return accuracy

# =============================================================================
# YOUR CODING Exam:
# =============================================================================
# HINT: The original function signature was:
# def train_model(reg_rate, X_train, X_test, y_train, y_test):
# TODO: David Review Complete
@component(  # TODO: Lab 5.8.1.2a - WHERE: @component decorator replaces def
    base_image=BASE_IMAGE,  # TODO: Lab 5.8.1.2b - WHAT: Container execution vs local Python
    packages_to_install=[  # TODO: Lab 5.8.1.2c - WHERE: Explicit dependencies vs local imports
        "google-cloud-bigquery",  # TODO: Lab 5.8.1.2d - WHAT: Cloud data access vs pandas
        "scikit-learn",  # TODO: Lab 5.8.1.2e - WHERE: Same sklearn but containerized
        "joblib",  # TODO: Lab 5.8.1.2f - WHAT: Model serialization for artifacts
        "pandas"  # TODO: Lab 5.8.1.2g - WHERE: DataFrame operations still needed
    ]
)
def train_model_op(  # TODO: Lab 5.8.1.2h - WHERE: Component function signature
        train_data: Input[artifact_types.BQTable],  # TODO: Lab 5.8.1.2i - WHAT: BQTable artifact vs X_train DataFrame
        output_model: Output[Model],  # TODO: Lab 5.8.1.2j - WHAT: Output[Model] vs return statement
        metrics: Output[Metrics],  # TODO: Lab 5.8.1.2k - WHAT: Structured metrics vs print()
        reg_rate: float,  # TODO: Lab 5.8.1.2l - WHERE: Same parameter, different data flow
        project_id: str,  # TODO: Lab 5.8.1.2m - WHAT: Cloud context vs local execution
        bq_location: str  # TODO: Lab 5.8.1.2n - WHAT: Regional data access parameter
) -> float:  # TODO: Lab 5.8.1.2o - WHERE: Return type for pipeline decisions
    """
    Train logistic regression model using BigQuery training data.
    Converted from train_model function in original train.py.

    Args:
        train_data: BigQuery table containing training data
        output_model: Output model artifact for pipeline consumption
        metrics: Training metrics for monitoring and evaluation
        reg_rate: Regularization rate (inverse of C parameter)
        project_id: Google Cloud project ID
        bq_location: BigQuery location/region

    Returns:
        float: Training accuracy for pipeline decision making
    """
    import re, os, shutil, joblib, logging
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)
    logging.info("[CONVERSION] Starting model training component")

    # TODO: Lab 5.8.2 - Data Source Translation: DataFrame input → BigQuery table parsing
    # EXPLORE: How train.py receives DataFrames vs pipeline components receive artifact URIs
    # TRANSLATE: Direct DataFrame access → URI parsing + BigQuery client
    uri = train_data.uri
    logging.info("[CONVERSION] Parsing BQ URI: %s", uri)
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BQ table from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    # TODO: Lab 5.8.3 - Data Loading Evolution: pandas.read_csv() → BigQuery client
    # EXPLORE: train.py loads data from memory vs component loads from BigQuery
    # UNDERSTAND: Same end result (DataFrame) but different data source
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    train_df = bq_client.query(query).to_dataframe()
    logging.info("[CONVERSION] Loaded %d training rows from BigQuery", len(train_df))

    # TODO: Lab 5.8.4 - Algorithm Consistency Exploration: Same sklearn code in both versions
    # INSTRUCTIONS: Find these exact same lines in train.py train_model function
    # WHAT STAYS THE SAME: Core ML algorithm logic remains identical

    # TODO: Lab 5.8.4.1 - ANSWER: Feature Engineering Consistency (WHAT: Same feature selection)
    # WHERE: Find these exact column names in train.py split_data function
    # WHY: Feature consistency ensures model behavior remains the same
    FEATURE_COLUMNS = ["Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
                       # TODO: Lab 5.8.4.1a - WHERE: Same features as train.py
                       "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree",
                       "Age"]  # TODO: Lab 5.8.4.1b - WHAT: Identical feature list
    X = train_df[FEATURE_COLUMNS]  # TODO: Lab 5.8.4.1c - WHERE: Same DataFrame indexing as train.py
    y = train_df["Diabetic"]  # TODO: Lab 5.8.4.1d - WHERE: Same target column as train.py

    # TODO: Lab 5.8.5 - Line-by-Line Algorithm Mapping: Identical sklearn training code
    # INSTRUCTIONS: Find these EXACT lines in train.py train_model function
    # WHAT'S IDENTICAL: Model initialization and training calls are 100% the same

    # TODO: Lab 5.8.5.1 - ANSWER: Find these exact lines in train.py train_model function:
    # model = LogisticRegression(C=1 / reg_rate, solver="liblinear")              # TODO: Lab 5.8.5.1a - WHERE: Exact same model initialization
    # model.fit(X_train, y_train)                                                 # TODO: Lab 5.8.5.1b - WHERE: Exact same fit() call
    model = LogisticRegression(C=1 / reg_rate,
                               solver="liblinear")  # TODO: Lab 5.8.5.1c - WHAT: Identical algorithm parameters
    model.fit(X, y)  # TODO: Lab 5.8.5.1d - WHAT: Same training call (X,y vs X_train,y_train)

    # TODO: Lab 5.8.5.2 - ANSWER: Accuracy Calculation (WHAT: Same evaluation method)
    # WHERE: Find model.score() call in train.py train_model function
    # WHY: Consistent evaluation ensures comparable results
    training_accuracy = model.score(X, y)  # TODO: Lab 5.8.5.2a - WHERE: Same score() method as train.py
    logging.info("[CONVERSION] Training accuracy: %.4f",
                 training_accuracy)  # TODO: Lab 5.8.5.2b - WHAT: Enhanced logging vs simple print()

    # TODO: Lab 5.8.6 - Model Persistence Translation: return model → artifact serialization
    # INSTRUCTIONS: Compare how train.py returns model vs component saves artifact
    # WHAT CHANGES: Direct return vs cloud artifact storage

    # TODO: Lab 5.8.6.1 - ANSWER: Original train.py model return (WHAT: Direct Python object return)
    # WHERE: Find "return model" statement in train.py train_model function
    # WHY: Functions return objects directly for immediate use

    # TODO: Lab 5.8.6.2 - ANSWER: Pipeline artifact storage (WHAT: Persistent cloud storage)
    # WHERE: joblib serialization and artifact path management
    # WHY: Distributed components need persistent, shareable model storage
    model_path = os.path.join(os.path.dirname(output_model.path),
                              "model.joblib")  # TODO: Lab 5.8.6.2a - WHAT: Artifact path generation
    joblib.dump(model, model_path)  # TODO: Lab 5.8.6.2b - WHAT: Model serialization for storage
    shutil.copy(model_path, output_model.path)  # TODO: Lab 5.8.6.2c - WHAT: Artifact path compliance

    # TODO: Lab 5.8.7 - Logging Evolution: print() statements → structured metrics
    # INSTRUCTIONS: Compare train.py print() vs pipeline metrics
    # WHAT IMPROVES: Simple console output vs structured, queryable metrics

    # TODO: Lab 5.8.7.1 - ANSWER: Original train.py logging (WHAT: Simple print statement)
    # WHERE: Find print(f"Model accuracy: {accuracy}") in train.py train_model function
    # WHY: Quick console output for immediate feedback

    # TODO: Lab 5.8.7.2 - ANSWER: Pipeline structured metrics (WHAT: Cloud-native observability)
    # WHERE: metrics.log_metric() calls for dashboard and monitoring integration
    # WHY: Enterprise monitoring, alerting, and historical tracking capabilities
    metrics.log_metric("training_accuracy", training_accuracy)  # TODO: Lab 5.8.7.2a - WHAT: Structured accuracy metric
    metrics.log_metric("regularization_rate", reg_rate)  # TODO: Lab 5.8.7.2b - WHAT: Parameter tracking for experiments
    metrics.log_metric("training_samples",
                       len(train_df))  # TODO: Lab 5.8.7.2c - WHAT: Data volume tracking for monitoring

    logging.info("[CONVERSION] Model stored at %s", output_model.path)
    return training_accuracy







# Model approval components (already completed for you)
@component(base_image=BASE_IMAGE)
def model_approved_op(model_accuracy: float, model_name: str):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("✅ Model '%s' approved with accuracy: %.4f", model_name, model_accuracy)

@component(base_image=BASE_IMAGE)
def model_rejected_op(model_accuracy: float, min_accuracy: float):
    import logging
    logging.basicConfig(level=logging.ERROR)
    logging.error("❌ Model rejected. Accuracy %.4f < %.2f", model_accuracy, min_accuracy)

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform"]
)
def register_model_op(
    project_id: str,
    region: str, 
    model_display_name: str,
    model_artifact: Input[Model],
    parent_model: str = ""
):
    from google.cloud import aiplatform
    import logging
    
    logging.basicConfig(level=logging.INFO)
    aiplatform.init(project=project_id, location=region)
    
    artifact_dir = model_artifact.uri.rsplit("/", 1)[0]
    
    upload_args = {
        "display_name": model_display_name,
        "artifact_uri": artifact_dir,
        "serving_container_image_uri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
        "sync": True
    }
    
    if parent_model:
        upload_args["parent_model"] = parent_model
    
    model = aiplatform.Model.upload(**upload_args)
    logging.info("Model registered: %s", model.resource_name)

# =============================================================================
# MAIN PIPELINE DEFINITION (already completed for you)
# =============================================================================
@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Exercise pipeline for train.py to Vertex AI conversion"
)
def diabetes_training_pipeline(
    project_id: str,
    region: str = "us-central1",
    model_display_name: str = "diabetes-classification-model",
    bq_dataset: str = "shared_bronze",
    bq_view: str = "diabetes_features_view",
    reg_rate: float = 0.01,
    min_accuracy: float = 0.70,
    parent_model: str = ""
):
    train_query = f"""
    SELECT Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness,
           SerumInsulin, BMI, DiabetesPedigree, Age, Diabetic
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(CONCAT(Pregnancies, PlasmaGlucose) AS STRING))), 10) < 8
    """
    
    test_query = f"""
    SELECT Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness,
           SerumInsulin, BMI, DiabetesPedigree, Age, Diabetic
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(CONCAT(Pregnancies, PlasmaGlucose) AS STRING))), 10) >= 8
    """
    
    bq_train_task = bigquery_query_job_op(
        project=project_id, 
        location=region, 
        query=train_query
    )
    
    bq_test_task = bigquery_query_job_op(
        project=project_id, 
        location=region, 
        query=test_query
    )
    
    # TODO: Uncomment and complete the train_task once you implement train_model_op

    train_task = train_model_op(
        train_data=bq_train_task.outputs["destination_table"],
        reg_rate=reg_rate,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.set_display_name("Train LogisticRegression")
    train_task.after(bq_train_task)

    
    # TODO: Uncomment the evaluation task once train_task is implemented

    eval_task = evaluate_model_op(
        test_data=bq_test_task.outputs["destination_table"],
        model=train_task.outputs["output_model"],
        min_accuracy=min_accuracy,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    eval_task.after(train_task)
    
    with dsl.If(eval_task.outputs["Output"] >= min_accuracy, name="pass-accuracy-threshold"):
        approved_task = model_approved_op(
            model_accuracy=eval_task.outputs["Output"],
            model_name=model_display_name
        )
        approved_task.after(eval_task)
        
        register_task = register_model_op(
            project_id=project_id,
            region=region,
            model_display_name=model_display_name,
            model_artifact=train_task.outputs["output_model"],
            parent_model=parent_model
        )
        register_task.after(approved_task)

    with dsl.If(eval_task.outputs["Output"] < min_accuracy, name="fail-accuracy-threshold"):
        rejected_task = model_rejected_op(
            model_accuracy=eval_task.outputs["Output"],
            min_accuracy=min_accuracy
        )
        rejected_task.after(eval_task)


# =============================================================================
# (optional - for advanced users)
# =============================================================================
# if __name__ == "__main__":
