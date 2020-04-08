class ImplementationError(Exception):
    """
    Exception for un-implemented methods in base classes.
    """

    def __init__(self, function_name, class_name):
        self.function_name = function_name
        self.class_name = class_name

    def __str__(self):
        return 'function {} needs to be implemented by {}'.format(self.function_name, self.class_name)
