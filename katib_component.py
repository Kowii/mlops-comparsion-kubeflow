from kfp.dsl import component
from typing import NamedTuple
from kfp import dsl

@component(
    base_image="python:3.10",
    packages_to_install=["kubernetes"]
)
def run_katib_tuning(
        dataset: dsl.Input[dsl.Dataset],
        experiment_name: str = "diabetes-hpo",
        namespace: str = "kubeflow"
) -> NamedTuple('Outputs', [('best_n_estimators', int), ('best_max_depth', int)]):
    from kubernetes import client, config
    import time
    dataset_uri = dataset.uri
    config.load_incluster_config()
    api = client.CustomObjectsApi()

    experiment_manifest = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {
            "name": experiment_name,
            "namespace": namespace
        },
        "spec": {
            "objective": {
                "type": "maximize",
                "goal": 0.99,
                "objectiveMetricName": "recall"
            },
            "algorithm": {
                "algorithmName": "random"
            },
            "parallelTrialCount": 1,
            "maxTrialCount": 5,
            "parameters": [
                {"name": "n_estimators", "parameterType": "int", "feasibleSpace": {"min": "50", "max": "300"}},
                {"name": "max_depth", "parameterType": "int", "feasibleSpace": {"min": "3", "max": "10"}}
            ],
            "trialTemplate": {
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {"name": "n_estimators", "description": "", "reference": "n_estimators"},
                    {"name": "max_depth", "description": "", "reference": "max_depth"}
                ],
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "template": {
                            "metadata": {
                                "annotations": {"sidecar.istio.io/inject": "false"}
                            },
                            "spec": {
                                "containers": [
                                    {
                                        "name": "training-container",
                                        "image": "localhost:32000/diabetes-train:latest",
                                        "imagePullPolicy": "IfNotPresent",
                                        "command": [
                                            "python", "/app/train_script.py",
                                            "--config_path", "s3://mlops-config/config.yaml",
                                            "--dataset_path", dataset_uri,
                                            "--n_estimators", "${trialParameters.n_estimators}",
                                            "--max_depth", "${trialParameters.max_depth}"
                                        ],
                                        "env": [
                                            {"name": "S3_ACCESS_KEY", "valueFrom": {
                                                "secretKeyRef": {"name": "s3-credentials", "key": "S3_ACCESS_KEY"}}},
                                            {"name": "S3_SECRET_KEY", "valueFrom": {
                                                "secretKeyRef": {"name": "s3-credentials", "key": "S3_SECRET_KEY"}}},
                                            {"name": "S3_ENDPOINT_URL", "valueFrom": {
                                                "secretKeyRef": {"name": "s3-credentials", "key": "S3_ENDPOINT_URL"}}}
                                        ]
                                    }
                                ],
                                "restartPolicy": "Never"
                            }
                        }
                    }
                }
            }
        }
    }

    print(f"Applying experiment to k8s cluster: {experiment_name}")
    try:
        api.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v1beta1",
            namespace=namespace,
            plural="experiments",
            body=experiment_manifest,
        )
    except client.exceptions.ApiException as e:
        if e.status == 409:  #
            print("Replacing existing experiment with new...")
            api.delete_namespaced_custom_object(
                group="kubeflow.org", version="v1beta1", namespace=namespace,
                plural="experiments", name=experiment_name
            )
            time.sleep(5)
            api.create_namespaced_custom_object(
                group="kubeflow.org", version="v1beta1", namespace=namespace,
                plural="experiments", body=experiment_manifest,
            )
        else:
            raise e

    print("Waiting for experiment to finish...")
    while True:
        exp = api.get_namespaced_custom_object(
            group="kubeflow.org",
            version="v1beta1",
            namespace=namespace,
            plural="experiments",
            name=experiment_name
        )

        status = exp.get("status", {})
        conditions = status.get("conditions", [])

        is_finished = False
        for cond in conditions:
            if cond.get("type") in ["Succeeded", "Failed"] and cond.get("status") == "True":
                print(f"Experiment ended successfully: {cond.get('type')}")
                is_finished = True
                break

        if is_finished:
            break

        print("Waiting...")
        time.sleep(15)

    exp = api.get_namespaced_custom_object(
        group="kubeflow.org", version="v1beta1", namespace=namespace,
        plural="experiments", name=experiment_name
    )

    best_trial = exp["status"]["currentOptimalTrial"]
    best_params = {p["name"]: p["value"] for p in best_trial["parameterAssignments"]}

    print(f"Optimized parameters: {best_params}")

    return (int(best_params["n_estimators"]), int(best_params["max_depth"]))
