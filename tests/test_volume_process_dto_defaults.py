from core.dto import VolumeProcessDTO
from config import SEGMENTATION_PROFILE_DEFAULT, SEGMENTATION_SPLIT_MODE_DEFAULT


def test_volume_process_dto_uses_segmentation_defaults():
    dto = VolumeProcessDTO()
    assert dto.segmentation_profile == SEGMENTATION_PROFILE_DEFAULT
    assert dto.split_mode == SEGMENTATION_SPLIT_MODE_DEFAULT


def test_volume_process_dto_from_dict_uses_segmentation_defaults():
    dto = VolumeProcessDTO.from_dict({})
    assert dto.segmentation_profile == SEGMENTATION_PROFILE_DEFAULT
    assert dto.split_mode == SEGMENTATION_SPLIT_MODE_DEFAULT
