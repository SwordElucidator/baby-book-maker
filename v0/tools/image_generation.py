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
    illustration_prompts: list[dict[str, str | list[str]]] = Field(
        ...,
        description="""List[{"prompt": str, "character_names": list[str]}] to generate images (one per page, same as the size of pages)"""
    )
    character_designs: list[dict[str, str]] = Field(
        ...,
        description="""All character designs List[{"name": str, "design": str}] copied from previous ArtDirection result"""
    )
    color_palette: str = Field(
        ...,
        description="The color palette for the whole picture book in 8 words"
    )
    art_style: str = Field(
        ...,
        description="The art style for the whole picture book in 6 words"
    )
    # width: int = Field(
    #     default=1280,
    #     description="Width of the generated images in pixels"
    # )
    # height: int = Field(
    #     default=720,
    #     description="Height of the generated images in pixels"
    # )

class BatchImageGenerationTool(BaseTool):
    name: str = "Batch Image Generation Tool"
    description: str = "Generates images from text descriptions using the FLUX.1 model"
    args_schema: Type[BaseModel] = ImageGenerationSchema
    model_name: str = "fal-ai/flux/dev" # "fal-ai/flux/schnell"

    def _run(self, illustration_prompts: list[dict[str, str | list[str]]], character_designs: list[dict[str, str]], color_palette: str, art_style: str) -> list[str]:
        """
        Generate multiple images based on text prompts.
        
        Args:
            illustration_prompts (List[{"prompt": str, "character_names": list[str]}]): List of prompts to generate images from
            character_designs (list[{"name": str, "design": str}]): List of character designs
            color_palette (str): The color palette for the picture book in 8 words
            art_style (str): The art style for the picture book in 6 words
            width (int): Width of the generated images in pixels
            height (int): Height of the generated images in pixels
            
        Returns:
            list[str]: List of URLs to the generated image files
        """
        results = []
        for ill_prompt in illustration_prompts:
            needed_character_designs = [
                f'{character_design["name"]}: {character_design["design"]}' for character_design in character_designs 
                if character_design["name"] in ill_prompt['character_names']
                ]
            prompt = f'{ill_prompt["prompt"]}\ncolor palette: {color_palette}, art style: {art_style}\n' + "\n".join(needed_character_designs)
            
            result = fal_client.subscribe(
                self.model_name,
                arguments={
                    "prompt": prompt,
                    "image_size": {
                        "width": 720,  # FIXME fixed for now
                        "height": 1280
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
    
    data = {
    "illustration_prompts": [
        {
            "prompt": "A welcoming fire station interior, warm lighting, young boy James in mini firefighter outfit with oversized boots and helmet, standing with his excited classmates. Captain Mike in full firefighter uniform demonstrating equipment. Soft edges, child-friendly style, elevated angle view. Bright red fire truck visible in background. Colors: bright red #FF3B3B, navy blue #1B365D, warm lighting.",
            "character_names": [
                "James",
                "Captain Mike"
            ]
        },
        {
            "prompt": "Gleaming red fire truck under morning sunlight, highly detailed with chrome accents. Spot the Dalmatian sitting proudly beside it, tail wagging. Clean white station walls, soft shadows. Dynamic composition with truck as focal point. Colors: bright red #FF3B3B, clean white #FFFFFF, sky blue #87CEEB background.",
            "character_names": [
                "Spot"
            ]
        },
        {
            "prompt": "Dramatic moment in fire station, siren lights flashing red and white. Children covering ears, James looking excited. Captain Mike rushing in with determined expression. Spot alert and attentive. Dynamic lighting from emergency lights. Colors: bright red #FF3B3B, clean white #FFFFFF, navy blue #1B365D.",
            "character_names": [
                "James",
                "Captain Mike",
                "Spot"
            ]
        },
        {
            "prompt": "Interior of fire station, Captain Mike in heroic pose pointing forward, James looking determined. Emergency lights casting dramatic shadows. Spot ready by the fire truck. Warm lighting with emphasis on characters' expressions. Colors: navy blue #1B365D, bright red #FF3B3B, silver #C0C0C0.",
            "character_names": [
                "James",
                "Captain Mike",
                "Spot"
            ]
        },
        {
            "prompt": "Residential street scene, two-story house with dark smoke billowing from windows. Fire truck arriving, James and Captain Mike visible in truck. Dramatic but not scary atmosphere. Soft edges maintain child-friendly appearance. Colors: soft gray #A4A4A4 smoke, sky blue #87CEEB, bright red #FF3B3B.",
            "character_names": [
                "James",
                "Captain Mike"
            ]
        },
        {
            "prompt": "Dynamic scene of firefighters spraying water on house. Beautiful water effects with rainbow highlights. James watching in awe. Dramatic but safe feeling composition. Steam and water spray creating interesting patterns. Colors: bright red #FF3B3B, clean white #FFFFFF, sky blue #87CEEB.",
            "character_names": [
                "James",
                "Captain Mike"
            ]
        },
        {
            "prompt": "Close-up shot of James helping Captain Mike hold the water hose, both wearing firefighter gear. James's determined expression visible under slightly-too-big helmet. Water spray creating dramatic effects. Warm encouraging smile from Captain Mike. Colors: navy blue #1B365D, bright red #FF3B3B, clean white #FFFFFF.",
            "character_names": [
                "James",
                "Captain Mike"
            ]
        },
        {
            "prompt": "Thompson family hugging in front of their house, looking relieved and grateful. James standing proudly nearby with Captain Mike's hand on his shoulder. Spot sitting alertly. Warm evening lighting, smoke clearing. Colors: warm beige #FFE5B4, bright red #FF3B3B, sky blue #87CEEB.",
            "character_names": [
                "James",
                "Captain Mike",
                "Spot",
                "Thompson Family"
            ]
        },
        {
            "prompt": "Captain Mike kneeling beside James at eye level, placing a junior firefighter badge on James's chest. Both smiling warmly. Other firefighters and Spot in background looking proud. Warm lighting emphasizing the moment. Colors: navy blue #1B365D, bright red #FF3B3B, silver #C0C0C0.",
            "character_names": [
                "James",
                "Captain Mike",
                "Spot"
            ]
        },
        {
            "prompt": "Celebratory scene inside fire station, James, Captain Mike, firefighters, and Spot gathered around a table with a cake. Fire truck in background. Warm, joyful lighting. Everyone smiling and celebrating. Colors: bright red #FF3B3B, clean white #FFFFFF, warm yellow #FFD700.",
            "character_names": [
                "James",
                "Captain Mike",
                "Spot"
            ]
        }
    ],
    "character_designs": [
        {
            "name": "James",
            "design": "A small boy (5-6 years old) with round, friendly features and rosy cheeks. Short, tousled brown hair. Wears a miniature firefighter outfit with oversized boots and helmet that slides slightly over his eyes for endearing effect. Proportions should be child-like with a larger head-to-body ratio. His expressions should be enthusiastic and determined."
        },
        {
            "name": "Captain Mike",
            "design": "Tall, broad-shouldered but not intimidating figure with a warm smile and gentle eyes. Salt-and-pepper hair visible under his helmet. Traditional firefighter uniform with visible badges. Friendly crow's feet around eyes when smiling. Movements and poses should show both authority and gentleness."
        },
        {
            "name": "Spot",
            "design": "Classic Dalmatian with exaggerated spots and expressive eyes. Slightly larger than average size to appear protective but friendly. Red collar with a shiny badge. Tail should be actively wagging in most scenes. Playful poses that mirror the emotional tone of each scene."
        },
        {
            "name": "Thompson Family",
            "design": "Mom: Medium height with warm expression and casual clothing. Dad: Tall with glasses and kind face. Little Sister: Around 3 years old with pigtails and a favorite stuffed animal. All characters should have simple but distinct features and matching worried expressions during the emergency that turn relieved and grateful after rescue."
        }
    ],
        "color_palette": "bright red navy blue warm yellow clean white soft gray warm beige forest green sky blue",
        "art_style": "child-friendly rounded soft elevated-angle",
    }
    
    result = tool._run(
        illustration_prompts=data["illustration_prompts"],
        character_designs=data["character_designs"],
        color_palette=data["color_palette"],
        art_style=data["art_style"]
    )
    
    print(result)
