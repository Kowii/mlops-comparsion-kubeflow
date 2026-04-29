# mlops_pipeline.py

import data_component, train_component, deploy_component
from kfp.dsl import pipeline
from kfp import kubernetes

@pipeline(
    name='mlops_pipeline',
    description="E2E pipeline for training classification model"
)
def mlops_pipeline(
        config_path: str = "s3://mlops-config/config.yaml",
        pipeline_n_estimators: int = 0,
        pipeline_max_depth: int = 0,
        pipeline_algorithm: str = ""
):
    # etap przygotowania danych
    data_prep_task = data_component.prepare(
        config_path=config_path,
    )
    kubernetes.use_secret_as_env(data_prep_task, secret_name='s3-credentials', secret_key_to_env={
            'S3_ACCESS_KEY': 'S3_ACCESS_KEY',
            'S3_SECRET_KEY': 'S3_SECRET_KEY',
            'S3_ENDPOINT_URL': 'S3_ENDPOINT_URL'
        })
    data_prep_task.set_caching_options(False)

    # etap trenowania modelu
    model_train_task = train_component.train(
        config_path=config_path,
        input_dataset=data_prep_task.outputs['output_dataset'],
        override_n_estimators=pipeline_n_estimators,
        override_max_depth=pipeline_max_depth,
        override_algorithm=pipeline_algorithm
    )
    model_train_task.set_caching_options(False)
    kubernetes.use_secret_as_env(
        task=model_train_task,
        secret_name='s3-credentials',
        secret_key_to_env={'S3_ACCESS_KEY': 'S3_ACCESS_KEY', 'S3_SECRET_KEY': 'S3_SECRET_KEY',
                           'S3_ENDPOINT_URL': 'S3_ENDPOINT_URL'}
    )

    deploy_task = deploy_component.deploy(
        model=model_train_task.outputs['output_model']
    )
    deploy_task.set_caching_options(False)

if __name__ == '__main__':
    from kfp.compiler import Compiler

    Compiler().compile(
        pipeline_func=mlops_pipeline,
        package_path="mlops_pipeline.yaml"
    )
