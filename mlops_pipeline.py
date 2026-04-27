# mlops_pipeline.py

import data_component, train_component
from kfp.dsl import pipeline
from kfp import kubernetes

@pipeline(
    name='mlops_pipeline',
    description="E2E pipeline for training classification model"
)
def mlops_pipeline(
        config_path: str = "s3://mlops-config/config.yaml"
):
    # etap przygotowania danych
    data_prep_task = data_component.prepare(
        config_path=config_path,
    )
    kubernetes.use_secret_as_env(data_prep_task, secret_name='s3_credentials', secret_key_to_env={
            'S3_ACCESS_KEY': 'S3_ACCESS_KEY',
            'S3_SECRET_KEY': 'S3_SECRET_KEY',
            'S3_ENDPOINT_URL': 'S3_ENDPOINT_URL'
        })

    # etap trenowania modelu
    model_train_task = train_component.train(
        config_path=config_path,
        input_dataset=data_prep_task.outputs['output_dataset']
    )
    kubernetes.use_secret_as_env(
        task=model_train_task,
        secret_name='s3-credentials',
        secret_key_to_env={'S3_ACCESS_KEY': 'S3_ACCESS_KEY', 'S3_SECRET_KEY': 'S3_SECRET_KEY',
                           'S3_ENDPOINT_URL': 'S3_ENDPOINT_URL'}
    )


if __name__ == '__main__':
    from kfp.compiler import Compiler

    Compiler().compile(
        pipeline_func=mlops_pipeline,
        package_path="mlops_pipeline.yaml"
    )