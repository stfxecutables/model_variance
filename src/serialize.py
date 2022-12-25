from abc import ABC, abstractclassmethod, abstractmethod, abstractstaticmethod
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class FileJSONable(ABC, Generic[T]):
    """Mixin for classes that can be written and read from a JSON
    representation"""

    @abstractmethod
    def to_json(self, path: Path) -> None:
        ...

    @abstractstaticmethod
    def from_json(path: Path) -> T:
        ...


class DirJSONable(ABC, Generic[T]):
    """Mixin for classes that can be written and read from a number of JSON
    representations"""

    @abstractmethod
    def to_json(self, root: Path) -> None:
        ...

    @abstractclassmethod
    def from_json(cls, root: Path) -> T:
        ...
