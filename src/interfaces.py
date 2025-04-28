from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")

@runtime_checkable
class SupportsFromConfig(Protocol):
    @staticmethod
    def from_config(**config) -> T: ...
