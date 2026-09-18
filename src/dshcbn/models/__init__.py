"""Model registry for the primary and ablation encoders."""

from .residual_cnn import CONCEPT_SPECS


def get_model_class(encoder: str):
    """Return the exact model class for a named encoder experiment."""
    if encoder == "residual_cnn":
        from .residual_cnn import FocusedDSHCBN
        return FocusedDSHCBN
    if encoder == "unet":
        from .unet import FocusedDSHCBN
        return FocusedDSHCBN
    raise ValueError(f"Unknown encoder: {encoder}")


def build_model(encoder: str, dropout: float = 0.25, **kwargs):
    """Construct an encoder variant while preserving primary-model defaults."""
    cls = get_model_class(encoder)
    if encoder == "unet":
        accepted = {k: kwargs[k] for k in ("encoder_drop", "encoder_norm", "encoder_final_channels") if k in kwargs}
        return cls(dropout=dropout, **accepted)
    return cls(dropout=dropout)


__all__ = ["CONCEPT_SPECS", "build_model", "get_model_class"]
