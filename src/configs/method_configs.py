# Copyright 2025 Sony Group Corporation.
# All rights reserved.
#
# Licenced under the License reported at
#
#     https://github.com/LTTM/MultimodalStudio/LICENSE.txt (the "License").
#
# This code is a modified version of the original code available at
#
#     https://github.com/autonomousvision/sdfstudio (commit 370902a10dbef08cb3fe4391bd3ed1e227b5c165)
#
# At the moment of this file creation, the original code is licensed under the Apache License, Version 2.0;
# You may obtain a copy of the Apache License, Version 2.0, at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# See the License for the specific language governing permissions and limitations under the License.
#
# Author: Federico Lincetto, Ph.D. Student at the University of Padova

"""
Predefined method configurations.
Each configuration specifies a rigid pipeline, with a fixed set of modules and of init values.
"""

import copy

from configs.configs import TrainerConfig, LoggingConfig, LocalWriterConfig
from data.datamanager import DataManagerConfig
from data.datasets import RawMultimodalAlignedDatasetConfig, MultimodalAlignedDatasetConfig, \
    MultimodalUnalignedDatasetConfig, RawMultimodalUnalignedDatasetConfig
from cameras.camera_optimizers import CameraOptimizerConfig
from cameras.pixel_samplers import UniformPixelSamplerConfig
from engine.evaluator import EvaluatorConfig, RawEvaluatorConfig
from engine.optimizers import AdamWOptimizerConfig
from engine.schedulers import MultiStepWarmupSchedulerConfig, CurvatureLossWarmUpSchedulerConfig
from evaluator_components.mesh_extractors import MeshExtractorConfig
from evaluator_components.pose_extractor import PoseExtractorConfig
from field_components.field_heads import PolarizationHeadConfig, ModalityHeadConfig
from field_components.encodings import HashEncodingConfig, NeRFEncodingConfig, SHEncodingConfig
from field_components.feature_structures import FeatureGridConfig, FeatureGridAndMLPConfig
from field_components.mlp import MLPConfig
from field_components.spatial_distortions import SceneContractionConfig
from fields.nerf_field import NeRFFieldConfig
from fields.radiance_field import RadianceFieldConfig
from fields.surface_field import SDFFieldConfig
from model_components.background_model import BackgroundModelConfig
from model_components.volume_rendering import NeuSDensityConfig, NeuSVolumeRenderingConfig
from model_components.losses import LossManagerConfig, EikonalLossConfig, CurvatureLossConfig, LossConfig, \
    SkipSaturationLossConfig
from model_components.radiance_model import RadianceModelConfig
from model_components.ray_samplers import LinearDisparitySamplerConfig, NeuSSamplerConfig
from model_components.renderers import RendererConfig, RadianceRenderer
from model_components.surface_model import SurfaceModelConfig
from models.base_model import BaseModelConfig
from pipelines.base_pipeline import BasePipelineConfig
from pipelines.raw_pipeline import RawPipelineConfig

method_configs = {}

# Method that uses multi-resolution hash grids as learnable fields for both surface and radiance models.
# It requires demosaicked frames.
method_configs['grid'] = TrainerConfig(
    method_name='grid',
    max_num_iterations=100000,
    steps_per_eval_batch=100,
    steps_per_eval_image=1000,
    steps_per_eval_all_images=25000,
    steps_per_export_mesh=5000,
    steps_per_export_poses=5000,
    steps_per_save=5000,
    mixed_precision=False,
    matmul_precision="high",
    save_only_latest_checkpoint=True,
    pipeline=BasePipelineConfig(
        datamanager=DataManagerConfig(
            dataset_class=MultimodalAlignedDatasetConfig(),
            pixel_sampler=UniformPixelSamplerConfig(
                num_rays_per_modality=32
            ),
            camera_optimizer=CameraOptimizerConfig(
                mode="off",
            ),
        ),
        model=BaseModelConfig(
            ray_sampler=NeuSSamplerConfig(
                num_samples=32,
                num_samples_importance=32,
            ),
            background_ray_sampler=LinearDisparitySamplerConfig(),
            surface_model=SurfaceModelConfig(
                use_numerical_gradients=True,
                surface_field=SDFFieldConfig(
                    field=FeatureGridAndMLPConfig(
                        feature_grid=FeatureGridConfig(
                            encoding=HashEncodingConfig(
                                max_res=1024,
                            ),
                            coarse_to_fine=True,
                            radius=1,
                        ),
                        mlp_head=MLPConfig(
                            num_layers=3,
                            activation="Softplus",
                            activation_params={
                                "beta": 100,
                            },
                            out_activation="None",
                            geometric_init=True,
                            weight_norm=True,
                        ),
                    ),
                    use_position_encoding=True,
                    position_encoding=NeRFEncodingConfig(
                        num_frequencies=6,
                        min_freq_exp=0.0,
                        max_freq_exp=5,
                        include_input=True,
                    ),
                ),
                volume_rendering=NeuSVolumeRenderingConfig(
                    density_fn=NeuSDensityConfig()
                ),
                compute_hessian=True,
            ),
            radiance_model=RadianceModelConfig(
                radiance_field=RadianceFieldConfig(
                    base_field=FeatureGridAndMLPConfig(
                        feature_grid=FeatureGridConfig(
                            encoding=HashEncodingConfig(
                                max_res=1024,
                            ),
                            coarse_to_fine=True,
                            radius=1,
                        ),
                        mlp_head=MLPConfig(
                            num_layers=3,
                            hidden_dim=256,
                            out_activation="ReLU",
                            weight_norm=True,
                        ),
                    ),
                ),
                radiance_feature_dim=256,
                modality_heads={
                    "rgb": ModalityHeadConfig(
                        field=MLPConfig(
                            num_layers=3,
                            hidden_dim=64,
                            out_activation="Sigmoid",
                            weight_norm=True,
                        ),
                    ),
                    "infrared": ModalityHeadConfig(
                        field=MLPConfig(
                            num_layers=3,
                            hidden_dim=64,
                            out_activation="Sigmoid",
                            weight_norm=True,
                        ),
                    ),
                    "mono": ModalityHeadConfig(
                        field=MLPConfig(
                            num_layers=3,
                            hidden_dim=64,
                            out_activation="Sigmoid",
                            weight_norm=True,
                        ),
                    ),
                    "polarization": PolarizationHeadConfig(
                        field=MLPConfig(
                            num_layers=3,
                            hidden_dim=256,
                            out_activation="None",
                            weight_norm=True,
                        ),
                    ),
                    "multispectral": ModalityHeadConfig(
                        field=MLPConfig(
                            num_layers=3,
                            hidden_dim=64,
                            out_activation="Sigmoid",
                            weight_norm=True,
                        ),
                    ),
                },
                use_direction_encoding=True,
                direction_encoding=SHEncodingConfig(
                    degree=4,
                ),
                use_reflection_direction=True,
                use_n_dot_v=True,
            ),
            background_model=BackgroundModelConfig(
                background_field=NeRFFieldConfig(
                    base_field=MLPConfig(
                        activation="ReLU",
                        hidden_dim=256,
                        num_layers=4,
                        out_activation="ReLU",
                        weight_norm=True,
                    ),
                    head_field=MLPConfig(
                        num_layers=4,
                        out_activation="ReLU",
                        weight_norm=True,
                    ),
                    use_position_encoding=True,
                    position_encoding=NeRFEncodingConfig(
                        num_frequencies=6,
                        min_freq_exp=0.0,
                        max_freq_exp=5,
                        include_input=True,
                    ),
                    use_direction_encoding=True,
                    direction_encoding=NeRFEncodingConfig(
                        num_frequencies=4,
                        min_freq_exp=0.0,
                        max_freq_exp=3,
                        include_input=True,
                    ),
                ),
                radiance_feature_dim=128,
                modality_heads={
                    "polarization": PolarizationHeadConfig()
                },
                spatial_distortion=SceneContractionConfig(
                    order=float("inf"),
                )
            ),
            renderer=RendererConfig(
                renderers={
                    "rgb": RadianceRenderer,
                    "mono": RadianceRenderer,
                    "multispectral": RadianceRenderer,
                    "infrared": RadianceRenderer,
                    "polarization": RadianceRenderer,
                }
            ),
        ),
        loss_manager=LossManagerConfig(
            radiance_losses={
                "rgb": LossConfig(),
                "mono": LossConfig(),
                "multispectral": LossConfig(),
                "infrared": LossConfig(),
                "polarization": SkipSaturationLossConfig(
                    saturation_threshold=0.9980,
                ),
            },
            geometry_losses={
                "eikonal_loss": EikonalLossConfig(),
                "curvature_loss": CurvatureLossConfig(
                    scheduler=CurvatureLossWarmUpSchedulerConfig(
                        warm_up_ratio=0.1,
                    )
                ),
            }
        ),
        optimizers={
            "fields": {
                "optimizer": AdamWOptimizerConfig(lr=1e-3, weight_decay=0.01, eps=1e-15),
                "scheduler": MultiStepWarmupSchedulerConfig(warm_up_ratio=0.1, milestones=[0.5, 0.75, 0.9], gamma=0.4),
            },
            "camera_poses": {
                "optimizer": AdamWOptimizerConfig(lr=1e-4, weight_decay=0.01, eps=1e-15),
                "scheduler": MultiStepWarmupSchedulerConfig(warm_up_ratio=0.1, milestones=[0.5, 0.75, 0.9], gamma=0.4),
            },
        },
        evaluator=EvaluatorConfig(
            eval_num_rays_per_chunk=1024,
            rendering_scale=0.25,
            roi_only=True,
            mesh_extractor=MeshExtractorConfig(
                marching_cube_threshold=0.0,
                gt_scale=False,
            ),
            pose_extractor=PoseExtractorConfig(
                gt_scale=False,
                colors={
                    "rgb": "green",
                    "infrared": "red",
                    "multispectral": "blue",
                    "mono": "black",
                    "polarization": "magenta"
                },
            ),
        ),
    ),
    logging=LoggingConfig(
        steps_per_log=100,
        steps_per_flush_buffer=100,
        max_buffer_size=20,
        local_writer=LocalWriterConfig(
            enable=True,
            max_log_size=10,
        ),
        enable_profiler=False,
    ),
)

# Method that uses MLPs as learnable fields for both surface and radiance models.
# It requires demosaicked frames.
method_configs['mlp'] = copy.deepcopy(method_configs['grid'])
method_configs['mlp'].method_name = 'mlp'
method_configs['mlp'].pipeline.model.surface_model = SurfaceModelConfig(
    use_numerical_gradients=False,
    surface_field=SDFFieldConfig(
        field=MLPConfig(
            activation="Softplus",
            num_layers = 8,
            hidden_dim=256,
            activation_params={
                "beta": 100,
            },
            out_activation="None",
            skip_connections=(4,),
            geometric_init=True,
            weight_norm=True,
        ),
        use_position_encoding=True,
        position_encoding=NeRFEncodingConfig(
            num_frequencies=6,
            min_freq_exp=0.0,
            max_freq_exp=5,
            include_input=True,
        ),
    ),
    volume_rendering=NeuSVolumeRenderingConfig(
        density_fn=NeuSDensityConfig()
    ),
    compute_hessian=False,
)
method_configs['mlp'].pipeline.model.radiance_model = RadianceModelConfig(
    radiance_field=RadianceFieldConfig(
        base_field=MLPConfig(
            activation="ReLU",
            num_layers = 8,
            hidden_dim=256,
            out_activation="ReLU",
            skip_connections=(4,),
            weight_norm=True,
        ),
    ),
    radiance_feature_dim=256,
    modality_heads=copy.deepcopy(method_configs['grid'].pipeline.model.radiance_model.modality_heads),
    use_direction_encoding=True,
    direction_encoding=SHEncodingConfig(
        degree=4,
    ),
    use_reflection_direction=True,
    use_n_dot_v=True,
)
method_configs['mlp'].pipeline.loss_manager.geometry_losses = {
    "eikonal_loss": EikonalLossConfig(),
}

# Method that uses multi-resolution hash grids as learnable fields for both surface and radiance models.
# It requires raw (mosaicked) frames.
method_configs['grid_raw'] = copy.deepcopy(method_configs['grid'])
method_configs['grid_raw'].method_name = 'grid_raw'
method_configs['grid_raw'].pipeline = RawPipelineConfig(
    datamanager=DataManagerConfig(
        dataset_class=RawMultimodalAlignedDatasetConfig(),
        pixel_sampler=copy.deepcopy(method_configs['grid'].pipeline.datamanager.pixel_sampler),
        camera_optimizer=copy.deepcopy(method_configs['grid'].pipeline.datamanager.camera_optimizer)
    ),
    model=copy.deepcopy(method_configs['grid'].pipeline.model),
    loss_manager=copy.deepcopy(method_configs['grid'].pipeline.loss_manager),
    optimizers=copy.deepcopy(method_configs['grid'].pipeline.optimizers),
    evaluator=RawEvaluatorConfig(
        eval_num_rays_per_chunk=method_configs['grid'].pipeline.evaluator.eval_num_rays_per_chunk,
        rendering_scale=method_configs['grid'].pipeline.evaluator.rendering_scale,
        roi_only=method_configs['grid'].pipeline.evaluator.roi_only,
        mesh_extractor=copy.deepcopy(method_configs['grid'].pipeline.evaluator.mesh_extractor),
        pose_extractor=copy.deepcopy(method_configs['grid'].pipeline.evaluator.pose_extractor),
    ),
)

# Method that uses MLPs as learnable fields for both surface and radiance models.
# It requires raw (mosaicked) frames.
method_configs['mlp_raw'] = copy.deepcopy(method_configs['mlp'])
method_configs['mlp_raw'].method_name = 'mlp_raw'
method_configs['mlp_raw'].pipeline = RawPipelineConfig(
    datamanager=DataManagerConfig(
        dataset_class=RawMultimodalAlignedDatasetConfig(),
        pixel_sampler=copy.deepcopy(method_configs['mlp'].pipeline.datamanager.pixel_sampler),
        camera_optimizer=copy.deepcopy(method_configs['mlp'].pipeline.datamanager.camera_optimizer)
    ),
    model=copy.deepcopy(method_configs['mlp'].pipeline.model),
    loss_manager=copy.deepcopy(method_configs['mlp'].pipeline.loss_manager),
    optimizers=copy.deepcopy(method_configs['mlp'].pipeline.optimizers),
    evaluator=RawEvaluatorConfig(
        eval_num_rays_per_chunk=method_configs['mlp'].pipeline.evaluator.eval_num_rays_per_chunk,
        rendering_scale=method_configs['mlp'].pipeline.evaluator.rendering_scale,
        roi_only=method_configs['mlp'].pipeline.evaluator.roi_only,
        mesh_extractor=copy.deepcopy(method_configs['mlp'].pipeline.evaluator.mesh_extractor),
        pose_extractor=copy.deepcopy(method_configs['mlp'].pipeline.evaluator.pose_extractor),
    ),
)

# Method based on 'grid'. It allows to load a different amount of frames per modality.
method_configs['grid_unbalanced'] = copy.deepcopy(method_configs['grid'])
method_configs['grid_unbalanced'].method_name = 'grid_unbalanced'
method_configs['grid_unbalanced'].pipeline.datamanager.dataset_class = MultimodalUnalignedDatasetConfig()

# Method based on 'grid_raw'. It allows to load a different amount of frames per modality.
method_configs['grid_raw_unbalanced'] = copy.deepcopy(method_configs['grid_raw'])
method_configs['grid_raw_unbalanced'].method_name = 'grid_raw_unbalanced'
method_configs['grid_raw_unbalanced'].pipeline.datamanager.dataset_class = RawMultimodalUnalignedDatasetConfig()

# Method based on 'grid'.
# It supervises only one random channel of each pixel, according to the specified channel probability.
method_configs['grid_decimated'] = copy.deepcopy(method_configs['grid'])
method_configs['grid_decimated'].method_name = 'grid_decimated'
method_configs['grid_decimated'].pipeline.loss_manager.radiance_losses["rgb"].per_channel_probability = [
    0.25, 0.5, 0.25
]
method_configs['grid_decimated'].pipeline.loss_manager.radiance_losses["multispectral"].per_channel_probability = [
    0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111
]
method_configs['grid_decimated'].pipeline.loss_manager.radiance_losses["polarization"].per_channel_probability = [
    0.25, 0.25, 0.25, 0.25
]

# Method based on 'grid_raw'. It allows to load a different amount of frames per modality and estimates the background
# with a multi-resolution hash grid instead of a mlp.
method_configs['grid_raw_grid_bg_unbalanced'] = copy.deepcopy(method_configs['grid_raw_unbalanced'])
method_configs['grid_raw_grid_bg_unbalanced'].method_name = 'grid_raw_grid_bg_unbalanced'
method_configs['grid_raw_grid_bg_unbalanced'].pipeline.model.background_model.background_field.base_field = FeatureGridAndMLPConfig(
    output_dim=256,
    feature_grid=FeatureGridConfig(
        encoding=HashEncodingConfig(
            max_res=1024,
        ),
        coarse_to_fine=True,
        radius=2,
    ),
    mlp_head=MLPConfig(
        num_layers=3,
        out_activation="ReLU"
    ),
)
method_configs['grid_raw_grid_bg_unbalanced'].pipeline.model.background_model.modality_heads = copy.deepcopy(method_configs['grid_raw'].pipeline.model.radiance_model.modality_heads)
method_configs['grid_raw_grid_bg_unbalanced'].pipeline.model.background_model.radiance_feature_dim = 256
