"""
Configuration module for application-wide constants and custom exception types.

This module centralizes shared configuration elements including exit codes and
custom exception classes that provide detailed error context. It serves as a
single source of truth for error handling and status codes throughout the
application.
"""


# %%
# Exit codes
class ExitCodes:
    SUCCESS_CODE = 0
    ERROR_CODE = 1
    EXCEPTION_CODE = 2
    SIGINT_CODE = 130


# %%
# Colours
class Colours:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


# %%
# Constants
class Constants:
    PATH_DEPTH = 5
    TIMEOUT_SEC = 30

    # Age group boundaries
    AGE_UNKNOWN = -1
    AGE_INFANT = 0
    AGE_TODDLER = 5
    AGE_CHILD = 10
    AGE_ADOLESCENT = 16
    AGE_YOUNG_ADULT = 18
    AGE_ADULT = 35
    AGE_MIDDLE_AGE = 50
    AGE_SENIOR = 65
    AGE_ELDERLY = 75
    AGE_OLDEST = 85


# %%
# Custom error types
class APIResponseError(Exception):
    """Raised when the API response cannot be parsed properly."""

    def __init__(self, response=None, parsing_error=None, message=None):
        self.response = response
        self.parsing_error = parsing_error
        self.message = message or f"Failed to parse API response: {parsing_error}"
        super().__init__(self.message)


class EnvironmentVariableError(Exception):
    """Raised when required environment variables are missing."""

    def __init__(self, missing_vars: list[str], message: str | None = None):
        self.missing_vars = missing_vars
        self.message = (
            message
            or f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        super().__init__(self.message)


class EmptyContainerError(Exception):
    """Raised when an Azure Storage container exists but contains no blobs."""

    def __init__(self, container_name, path=None, message=None):
        self.container_name = container_name
        self.path = path
        self.message = (
            message
            or f"Container '{container_name}' {f'at path {path}' if path else ''} "
            "exists but contains no blobs"
        )
        super().__init__(self.message)
