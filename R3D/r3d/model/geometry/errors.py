"""Errors raised by the canonical-frame geometry package."""


class GeometryError(ValueError):
    """Base class for deterministic geometry/configuration failures."""


class InvalidRotationError(GeometryError):
    pass


class InvalidTransformError(GeometryError):
    pass


class UnsupportedArrayTypeError(GeometryError, TypeError):
    pass


class UnsupportedDTypeError(GeometryError, TypeError):
    pass


class UnsupportedRepresentationError(GeometryError):
    pass


class SchemaDimensionError(GeometryError):
    pass


class SchemaOverlapError(GeometryError):
    pass


class SchemaCoverageError(GeometryError):
    pass


class MissingFieldError(GeometryError, KeyError):
    pass


class MissingQuaternionConventionError(GeometryError):
    pass


class MissingDeltaCompositionError(GeometryError):
    pass


class TransformPathNotFoundError(GeometryError):
    pass


class AmbiguousTransformPathError(GeometryError):
    pass


class InconsistentTransformCycleError(GeometryError):
    pass


class DuplicateTransformError(GeometryError):
    pass


class RuntimeTransformMissingError(GeometryError, KeyError):
    pass


class TimestampMismatchError(GeometryError):
    pass


class DoubleTransformError(GeometryError):
    pass


class FrameMetadataMismatchError(GeometryError):
    pass


class CheckpointFrameMismatchError(GeometryError):
    pass


class ProfileContractError(GeometryError):
    """A benchmark profile disagrees with its frozen native data contract."""

    pass
