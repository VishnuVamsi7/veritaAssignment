from __future__ import annotations

from typing import Iterable

from .elements import CanvasElement


class CanvasEngine:
    """In-memory canvas state manager with z-index aware ordering."""

    def __init__(self) -> None:
        self._elements: list[CanvasElement] = []

    def add_element(self, element: CanvasElement) -> int:
        """Add an element and return its index in sorted render order."""
        self._elements.append(element)
        self._sort_by_z_index()
        return self._elements.index(element)

    def move_element(self, element_index: int, x: float, y: float) -> CanvasElement:
        """Move an element to absolute x/y coordinates."""
        element = self._get_element(element_index)
        updated = element.model_copy(update={"x": x, "y": y})
        self._elements[element_index] = updated
        self._sort_by_z_index()
        return updated

    def update_element(self, element_index: int, **updates: object) -> CanvasElement:
        """Update one or more element properties."""
        element = self._get_element(element_index)
        updated = element.model_copy(update=updates)
        self._elements[element_index] = updated
        self._sort_by_z_index()
        return updated

    def delete_element(self, element_index: int) -> CanvasElement:
        """Delete an element by index and return the removed element."""
        self._get_element(element_index)
        return self._elements.pop(element_index)

    def get_element(self, element_index: int) -> CanvasElement:
        """Read an element by index."""
        return self._get_element(element_index)

    def list_elements(self) -> list[CanvasElement]:
        """Return a copy of elements in render order."""
        return list(self._elements)

    def set_elements(self, elements: Iterable[CanvasElement]) -> None:
        """Replace the full element list."""
        self._elements = list(elements)
        self._sort_by_z_index()

    def clear(self) -> None:
        """Clear all elements."""
        self._elements.clear()

    def _sort_by_z_index(self) -> None:
        # Stable sort preserves insertion order for equal z-index values.
        self._elements.sort(key=lambda element: element.z_index)

    def _get_element(self, element_index: int) -> CanvasElement:
        try:
            return self._elements[element_index]
        except IndexError as exc:
            raise IndexError(f"Element index {element_index} is out of range.") from exc
