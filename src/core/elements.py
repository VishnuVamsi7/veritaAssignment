from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ShapeKind(str, Enum):
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    LINE = "line"
    POLYGON = "polygon"


class CanvasElement(BaseModel):
    x: float = Field(..., description="X coordinate of top-left corner")
    y: float = Field(..., description="Y coordinate of top-left corner")
    width: float = Field(..., gt=0, description="Element width in pixels")
    height: float = Field(..., gt=0, description="Element height in pixels")
    z_index: int = Field(0, description="Render ordering index")
    color: str = Field("#000000", description="Element color in hex or named format")
    content: str = Field("", description="Text payload or source reference")


class TextElement(CanvasElement):
    element_type: Literal["text"] = "text"
    text_color: str = Field(
        "#000000", description="Text color in hex or named format"
    )


class ShapeElement(CanvasElement):
    element_type: Literal["shape"] = "shape"
    shape_type: ShapeKind = ShapeKind.RECTANGLE


class ImageElement(CanvasElement):
    element_type: Literal["image"] = "image"
