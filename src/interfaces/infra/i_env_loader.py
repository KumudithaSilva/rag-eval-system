from abc import ABC, abstractmethod


class IEnvLoader(ABC):

    @abstractmethod
    def load_env_variables(self) -> None:
        pass
