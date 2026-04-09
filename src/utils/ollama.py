import subprocess
import requests
import time
from infrastructure.logs.logger_singleton import Logger

logger = Logger("utils-file-ollama")


def ensure_ollama_running(timeout: float = 5.0, silent: bool = True):
    """
    Ensure the Ollama local server is running.

    Args:
        timeout (float): Maximum number of seconds to wait for Ollama to be ready.
        silent (bool): If True, suppresses Ollama logs by redirecting stdout and stderr to DEVNULL.

    Raises:
        RuntimeError: If Ollama fails to start within the specified timeout.
    """

    def is_running() -> bool:
        """
        Check if the Ollama server is running.

        Returns:
            bool: True if Ollama is reachable, False otherwise.
        """
        try:
            requests.get("http://localhost:11434", timeout=1)
            return True
        except requests.exceptions.RequestException:
            return False

    # Check if Ollama is already running
    if is_running():
        logger.info("Ollama is already running.")
        return

    logger.info("Ollama not running. Starting Ollama server...")

    # Prepare subprocess kwargs
    kwargs = {}
    if silent:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

    # Start Ollama in the background
    subprocess.Popen(["ollama", "serve"], **kwargs)

    # Wait until server is ready
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_running():
            logger.info("Ollama server is now running.")
            return
        time.sleep(0.5)

    logger.error("Ollama failed to start within the timeout period.")
    raise RuntimeError("Ollama failed to start within timeout")
