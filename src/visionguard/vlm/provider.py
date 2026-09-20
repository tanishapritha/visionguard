from abc import ABC, abstractmethod
from typing import Sequence

class VLMProvider(ABC):
    """Provider boundary for expensive video-language investigation."""
    @abstractmethod
    def investigate(self, frames: Sequence[str], prompt: str) -> dict:
        raise NotImplementedError
