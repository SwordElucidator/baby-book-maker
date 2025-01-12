import os
from typing import Any, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from huggingface_hub import InferenceClient
from dotenv import load_dotenv


class ImageGenerationSchema(BaseModel):
    """Input for ImageGenerationTool."""
    prompt: str = Field(
        ...,
        description="Text description of the image to generate"
    )
    width: int = Field(
        default=1280,
        description="Width of the generated image in pixels"
    )
    height: int = Field(
        default=720,
        description="Height of the generated image in pixels"
    )


load_dotenv()


client = InferenceClient(
    "black-forest-labs/FLUX.1-dev",
    token=os.getenv("HF_TOKEN")
)


class ImageGenerationTool(BaseTool):
    name: str = "Image Generation Tool"
    description: str = "Generates images from text descriptions using the FLUX.1 model"
    args_schema: Type[BaseModel] = ImageGenerationSchema

    def _run(self, prompt: str, width: int = 1280, height: int = 720) -> str:
        """
        Generate an image based on the text prompt.
        
        Args:
            prompt (str): Text description of the image to generate
            width (int): Width of the generated image in pixels
            height (int): Height of the generated image in pixels
            
        Returns:
            str: Path to the generated image file
        """
        try:
            # Generate image
            image = client.text_to_image(prompt, width=width, height=height)
            
            # Save image to file
            output_path = f"generated_images/{prompt[:30]}.jpg"
            os.makedirs("generated_images", exist_ok=True)
            image.save(output_path)
            
            return output_path
            
        except Exception as e:
            return f"Error generating image: {str(e)}"
