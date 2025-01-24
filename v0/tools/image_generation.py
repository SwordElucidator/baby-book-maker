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


load_dotenv()


client = InferenceClient(
    "black-forest-labs/FLUX.1-dev",
    token=os.getenv("HF_TOKEN")
)



class ImageGenerationSchema(BaseModel):
    """Input for BatchImageGenerationTool."""
    prompts: list[str] = Field(
        ...,
        description="List of text descriptions of images to generate"
    )
    width: int = Field(
        default=1280,
        description="Width of the generated images in pixels"
    )
    height: int = Field(
        default=720,
        description="Height of the generated images in pixels"
    )

class BatchImageGenerationTool(BaseTool):
    name: str = "Batch Image Generation Tool"
    description: str = "Generates images from text descriptions using the FLUX.1 model"
    args_schema: Type[BaseModel] = ImageGenerationSchema
    model_name: str = "fal-ai/flux/dev" # "fal-ai/flux/schnell"

    def _run(self, prompts: list[str], width: int = 1280, height: int = 720) -> list[str]:
        """
        Generate multiple images based on text prompts.
        
        Args:
            prompts (list[str]): List of text descriptions to generate images from
            width (int): Width of the generated images in pixels
            height (int): Height of the generated images in pixels
            
        Returns:
            list[str]: List of URLs to the generated image files
        """
        results = []
        for prompt in prompts:
            result = fal_client.subscribe(
                self.model_name,
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
            results.append(result['images'][0]['url'])
        return results




if __name__ == "__main__":
    # Test image generation
    tool = BatchImageGenerationTool()
    
    test_prompt = "A cute cartoon astronaut floating in space with stars and planets in the background, children's book style"
    
    print(f"Generating test image with prompt: {test_prompt}")
    result = tool._run(
        prompts=[test_prompt],
        width=1280,
        height=720
    )
    
    print(f"Image generated at: {result}")
