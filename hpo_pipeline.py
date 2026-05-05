# hpo_pipeline.py

import data_component, train_component, katib_component
from kfp.dsl import pipeline
from kfp import kubernetes
@pipeline(
    name='hpo_pipeline',
    description="Pipeline for training classification model with optimized hiperparameters using katib",
)
def hpo_pipeline(
        config_path: str = "s3://mlops-config/config.yaml",
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
    katib_task = katib_component.run_katib_tuning(
        dataset=data_prep_task.outputs['output_dataset'],
        experiment_name="diabetes-tuning"
    )
    kubernetes.add_pod_annotation(task=katib_task, annotation_key="sidecar.istio.io/inject", annotation_value="false")

    model_train_task = train_component.train(
        config_path=config_path,
        input_dataset=data_prep_task.outputs['output_dataset'],
        input_transformer=data_prep_task.outputs['output_transformer'],
        override_algorithm=pipeline_algorithm,
        override_n_estimators=katib_task.outputs['best_n_estimators'],
        override_max_depth=katib_task.outputs['best_max_depth']
    )
    model_train_task.set_caching_options(False)

    kubernetes.use_secret_as_env(
        task=model_train_task,
        secret_name='s3-credentials',
        secret_key_to_env={'S3_ACCESS_KEY': 'S3_ACCESS_KEY', 'S3_SECRET_KEY': 'S3_SECRET_KEY',
                           'S3_ENDPOINT_URL': 'S3_ENDPOINT_URL'}
    )

if __name__ == '__main__':
    from kfp.compiler import Compiler

    Compiler().compile(
        pipeline_func=hpo_pipeline,
        package_path="hpo_pipeline.yaml"
    )
