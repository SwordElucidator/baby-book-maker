
from typing import List
from litellm import BaseModel, Field


class ResearchResult(BaseModel):
    """Research result model"""
    theme: str = Field(..., description="The final theme for a picture book")
    educational_elements: List[str] = Field(..., description="The educational elements in the picture book")
    

class SinglePageInOutline(BaseModel):
    """Single page model"""
    core_vocabulary: str = Field(..., description="The core vocabulary word for the page")
    plot_point: str = Field(..., description="The plot point for the page")
    educational_elements: str = Field(..., description="The educational elements for the page")


class StoryOutline(BaseModel):
    """Story outline model"""
    title: str = Field(..., description="The title of the book")
    character_descriptions: List[str] = Field(..., description="The descriptions of the characters")
    pages: List[SinglePageInOutline] = Field(..., description="The pages in the book")


class PageContent(BaseModel):
    core_vocabulary_word: str = Field(..., description="The core vocabulary word for the page")
    content: str = Field(..., description="The content for the page. Must contains core vocabulary word.")


class PageContents(BaseModel):
    """Collection of page content"""
    pages: List[PageContent] = Field(..., description="The content for each page")
    

class CharacterDesign(BaseModel):
    """Character design model"""
    name: str = Field(..., description="The name of the character")
    design: str = Field(..., description="The visual design of the character")


class ArtDirection(BaseModel):
    """Art direction model"""
    character_designs: List[CharacterDesign] = Field(..., description="The character designs for the picture book")
    color_palette: str = Field(..., description="The color palette for the picture book")
    visual_style: str = Field(..., description="The visual style for the picture book")


class IllustrationPrompt(BaseModel):
    """Illustration prompt model"""
    prompt: str = Field(..., description="The prompt for the AI image generation")
    character_names: List[str] = Field(..., description="The character names needed for the page. Each name should be in the character design list.")


class IllustrationPrompts(BaseModel):
    """Illustrations model"""
    illustration_prompts: List[IllustrationPrompt] = Field(..., description="The AI image generation prompts for each page")


class Illustrations(BaseModel):
    """Illustration paths model"""
    image_size: str = Field(..., description="The consistent size of the illustration")
    illustration_paths: List[str] = Field(..., description="The paths to the generated illustrations for each page")



class TranslatedContents(BaseModel):
    """Translated content model"""
    pages: List[PageContent] = Field(..., description="The translated content for each page")
