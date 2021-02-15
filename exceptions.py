class MuZeroError(Exception):
    """
    Base class for exceptions in MuProver.
    """

    def __init__(self, message: str) -> None:
        self.message = message

    def __repr__(self) -> str:
        return self.message


class MuZeroImplementationError(MuZeroError):
    """
    Exception for un-implemented methods in base classes.
    """

    def __init__(self, function_name: str, class_name: str):
        super().__init__(message=f'function {function_name} needs to be implemented by {class_name}')
        self.function_name: str = function_name
        self.class_name: str = class_name


class MuZeroEnvironmentError(MuZeroError):
    """
    Exception for the environment classes.
    """

    def __init__(self, message: str):
        super().__init__(message=message)
