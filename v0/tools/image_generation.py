import os
from typing import Any, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import fal_client

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
           print(log["message"])


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
            result = fal_client.subscribe(
                "fal-ai/flux/schnell",
                arguments={
                    "prompt": prompt,
                    "image_size": {
                        "width": width,
                        "height": height
                    }
                },
                with_logs=True,
                on_queue_update=on_queue_update,
            )

            return result['url']
            
        except Exception as e:
            return f"Error generating image: {str(e)}"




if __name__ == "__main__":
    # Test image generation
    tool = ImageGenerationTool()
    
    test_prompt = "A cute cartoon astronaut floating in space with stars and planets in the background, children's book style"
    
    print(f"Generating test image with prompt: {test_prompt}")
    result = tool._run(
        prompt=test_prompt,
        width=1280,
        height=720
    )
    
    print(f"Image generated at: {result}")
