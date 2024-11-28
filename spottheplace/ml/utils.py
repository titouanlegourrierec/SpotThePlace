from typing import Tuple
from PIL import Image, ImageDraw


class AddMask:
    def __init__(self,
                 top_left_proportion: Tuple[float, float] = (0.25, 0.25),
                 low_left_proportion: Tuple[float, float] = (0.15, 0.15)) -> None:
        """
        Add rectangle masks in the top-left and bottom-left corners of the image.
            - top_left_proportion: tuple (width_ratio, height_ratio), where values are fractions of the image size.
            - low_left_proportion: tuple (width_ratio, height_ratio), where values are fractions of the image size.
        """
        self.top_left_proportion = top_left_proportion
        self.low_left_proportion = low_left_proportion

    def __call__(self, img: Image.Image) -> Image.Image:
        """
            - img: PIL Image
        """
        if not isinstance(img, Image.Image):
            raise TypeError("Input should be a PIL Image.")

        # Get image dimensions
        width, height = img.size

        # Compute top-left rectangle size based on proportions
        top_left_rect_width = int(width * self.top_left_proportion[0])
        top_left_rect_height = int(height * self.top_left_proportion[1])

        # Define top-left rectangle coordinates
        x0, y0 = 0, 0
        x1, y1 = top_left_rect_width, top_left_rect_height

        # Draw the top-left rectangle
        draw = ImageDraw.Draw(img)
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))

        # Compute bottom-left rectangle size based on proportions
        low_left_rect_width = int(width * self.low_left_proportion[0])
        low_left_rect_height = int(height * self.low_left_proportion[1])

        # Define bottom-left rectangle coordinates
        x0, y0 = 0, height - low_left_rect_height
        x1, y1 = low_left_rect_width, height

        # Draw the bottom-left rectangle
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))

        return img
