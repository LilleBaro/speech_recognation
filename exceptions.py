class AudioProcessingError(Exception):
    """Raised when an uploaded audio file cannot be processed (corrupt, empty, silent, unsupported)."""
