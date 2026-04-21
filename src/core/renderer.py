from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .elements import ImageElement, ShapeElement, TextElement
from .engine import CanvasEngine


class CanvasRenderer:
    """Renders canvas state from CanvasEngine using Pillow."""

    def __init__(
        self,
        engine: CanvasEngine,
        width: int = 800,
        height: int = 600,
        background_color: str = "#FFFFFF",
    ) -> None:
        self.engine = engine
        self.width = width
        self.height = height
        self.background_color = background_color
        self._font = ImageFont.load_default()
        self._last_image: Image.Image | None = None

    def render(self) -> Image.Image:
        """Render the current engine state onto an RGB image."""
        image = Image.new("RGB", (self.width, self.height), color=self.background_color)
        draw = ImageDraw.Draw(image)

        for element in self.engine.list_elements():
            x1 = int(element.x)
            y1 = int(element.y)
            x2 = int(element.x + element.width)
            y2 = int(element.y + element.height)
            bbox = [(x1, y1), (x2, y2)]

            if isinstance(element, TextElement):
                draw.rectangle(bbox, outline=element.color, width=1)
                draw.text(
                    (x1 + 2, y1 + 2),
                    element.content,
                    fill=element.text_color,
                    font=self._font,
                )
            elif isinstance(element, ShapeElement):
                draw.rectangle(bbox, fill=element.color, outline=element.color)
            elif isinstance(element, ImageElement):
                draw.rectangle(bbox, fill=element.color, outline=element.color)
                if element.content:
                    draw.text((x1 + 2, y1 + 2), element.content, fill="#000000", font=self._font)

        self._last_image = image
        return image

    def render_rgb_array(self) -> list[list[tuple[int, int, int]]]:
        """Render and return an 800x600-style RGB array as nested lists."""
        image = self.render()
        pixels = list(image.getdata())
        return [pixels[i : i + self.width] for i in range(0, len(pixels), self.width)]

    def save_to_png(self, path: str) -> None:
        """Save the latest render to a PNG file."""
        image = self._last_image or self.render()
        image.save(path, format="PNG")
