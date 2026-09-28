from xaytune.core.errors import XaytuneError


class CheckpointError(XaytuneError):
    """A checkpoint could not be safely saved or localized."""


class CheckpointCorruptionError(CheckpointError):
    """Manifest, referenced state, or bytes are missing or inconsistent."""


class CheckpointCompatibilityError(CheckpointError):
    """The requested state, data, or codec cannot safely resume this bundle."""
