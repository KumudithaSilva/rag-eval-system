from abc import abstractmethod


class IPipelineStep:
    """
    Interface for pipeline procedure.
    """

    @abstractmethod
    def run(self, data: dict) -> dict:
        """
        Run the pipeline step.

        Args:
            data (dict): Input data for the pipeline step.
        """
        pass
