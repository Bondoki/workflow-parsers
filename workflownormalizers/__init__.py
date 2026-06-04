from nomad.config.models.plugins import NormalizerEntryPoint
from pydantic import Field


class AflowVaspNormalizerEntryPoint(NormalizerEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from workflownormalizers.aflow.normalizer import AflowVaspNormalizer

        return AflowVaspNormalizer(**self.model_dump())


aflow_vasp_normalizer_entry_point = AflowVaspNormalizerEntryPoint(
    name='AFLOW_VASP_Normalizer',
    description='Normalizer for AFLOW entry point to enrich with VASP data.',
)
