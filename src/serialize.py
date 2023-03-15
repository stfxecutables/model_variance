from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, TypeVar

T = TypeVar("T")


class FileJSONable(ABC, Generic[T]):
    """Mixin for classes that can be written and read from a JSON
    representation"""

    @abstractmethod
    def to_json(self, path: Path) -> None:
        ...

    @abstractmethod
    def to_jsons(self) -> str:
        ...

    # see https://github.com/python/typeshed/issues/7134#issuecomment-1030570063
    @staticmethod
    @abstractmethod
    def from_json(path: Path) -> FileJSONable[T]:
        ...

    @staticmethod
    @abstractmethod
    def from_dict(hpdict: Dict[str, Any]) -> FileJSONable[T]:
        ...


class DirJSONable(ABC, Generic[T]):
    """Mixin for classes that can be written and read from a number of JSON
    representations"""

    @abstractmethod
    def to_json(self, root: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def from_json(cls, root: Path) -> T:
        ...


class SKOPable(ABC):
    @abstractmethod
    def to_skops(self, root: Path) -> None:
        ...
