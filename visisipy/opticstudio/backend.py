from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Literal

import zospy as zp
from zospy.zpcore import ZOS, OpticStudioSystem

from visisipy._backend import BaseBackend
from visisipy.opticstudio.models import BaseOpticStudioEye, OpticStudioEye

if TYPE_CHECKING:
    from visisipy import EyeModel


def initialize_opticstudio(
    mode: Literal["standalone", "extension"] = "standalone",
    zosapi_nethelper: str | None = None,
    opticstudio_directory: str | None = None,
) -> tuple[ZOS, OpticStudioSystem]:
    zos = zp.ZOS(zosapi_nethelper=zosapi_nethelper, opticstudio_directory=opticstudio_directory)
    oss = zos.connect(mode)

    return zos, oss


class OpticStudioBackend(BaseBackend):
    _zos: ZOS | None = None
    _oss: OpticStudioSystem | None = None
    _model: BaseOpticStudioEye | None = None

    @classmethod
    def initialize(
        cls,
        mode: Literal["standalone", "extension"] = "standalone",
        zosapi_nethelper: str | None = None,
        opticstudio_directory: str | None = None,
    ) -> None:
        cls._zos, cls._oss = initialize_opticstudio(
            mode=mode,
            zosapi_nethelper=zosapi_nethelper,
            opticstudio_directory=opticstudio_directory,
        )

    @classmethod
    def build_model(cls, model: EyeModel, replace_existing: bool = False, **kwargs) -> OpticStudioEye:
        if not replace_existing and cls._model is not None:
            cls.clear_model()

        opticstudio_eye = OpticStudioEye(model)
        opticstudio_eye.build(cls._oss, **kwargs)

        cls._model = opticstudio_eye

        return opticstudio_eye

    @classmethod
    def clear_model(cls) -> None:
        cls._oss.new(saveifneeded=False)
        cls._model = None

    @classmethod
    def save_model(cls, path: str | PathLike | None = None) -> None:
        if path is not None:
            cls._oss.save_as(path)
        else:
            cls._oss.save()

    @classmethod
    def disconnect(cls) -> None:
        cls._oss.close()
        cls._oss = None
        cls._zos.disconnect()
        cls._zos = None
