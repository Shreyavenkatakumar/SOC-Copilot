"""Custom exception types shared across the SOC Copilot Phase I pipeline."""


class SOCCopilotError(Exception):
    """Base exception for all SOC Copilot Phase I errors."""


class LogCollectionError(SOCCopilotError):
    """Raised when raw log collection or validation fails."""


class LogParsingError(SOCCopilotError):
    """Raised when Drain3 log parsing fails."""


class FeatureExtractionError(SOCCopilotError):
    """Raised when feature extraction fails."""


class ModelTrainingError(SOCCopilotError):
    """Raised when model training fails."""


class ModelInferenceError(SOCCopilotError):
    """Raised when model inference fails."""


class FusionError(SOCCopilotError):
    """Raised when the fusion engine fails to combine anomaly scores."""


class SeverityScoringError(SOCCopilotError):
    """Raised when severity scoring fails."""


class ExplainabilityError(SOCCopilotError):
    """Raised when SHAP explainability generation fails."""


class EvaluationError(SOCCopilotError):
    """Raised when evaluation metric computation fails."""
