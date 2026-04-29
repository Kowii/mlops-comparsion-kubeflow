# deploy_component.py
from kfp import dsl
from kfp.dsl import component


@component(
    base_image="python:3.10",
    packages_to_install=["kubernetes"]
)
def deploy(
        model: dsl.Input[dsl.Model],
        model_name: str = "diabetes-predictor",
        namespace: str = "kubeflow"
):
    """
    Wdraża model przy użyciu KServe InferenceService.
    """
    from kubernetes import client, config

    config.load_incluster_config()
    api = client.CustomObjectsApi()

    storage_uri = model.uri

    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
            "annotations": {
                "serving.kserve.io/secretKey": "s3-credentials"
            }
        },
        "spec": {
            "predictor": {
                "model": {
                    "storageUri": storage_uri,
                    "args": ["--enable_docs_url=True"],
                    "modelFormat": {
                        "name": "sklearn"
                        }
                }
            }
        }
    }

    print(f"Deploying model {model_name} from {storage_uri} to KServe...")

    try:
        api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            body=inference_service,
        )
        print("KServe InferenceService created successfully.")
    except Exception as e:
        print(f"Service might exist, trying to patch: {e}")
        api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=model_name,
            body=inference_service,
        )
        print("KServe InferenceService patched successfully.")
