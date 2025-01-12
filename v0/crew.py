import datetime
import json
import os
from typing import List
from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import litellm
from pydantic import BaseModel, Field

from v0.tools.image_generation import ImageGenerationTool


llm = LLM(model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0", temperature=0.8)


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
    content: str = Field(..., description="The content for the page")


class PageContents(BaseModel):
    """Collection of page content"""
    pages: List[PageContent] = Field(..., description="The content for each page")
    

class ArtDirection(BaseModel):
    """Art direction model"""
    art_direction: str = Field(..., description="The final art direction")


class IllustrationPrompts(BaseModel):
    """Illustrations model"""
    illustration_prompts: List[str] = Field(..., description="The AI image generation prompts for each page")


class Illustrations(BaseModel):
    """Illustration paths model"""
    image_size: str = Field(..., description="The consistent size of the illustration")
    illustration_paths: List[str] = Field(..., description="The paths to the generated illustrations for each page")



class TranslatedContents(BaseModel):
    """Translated content model"""
    pages: List[PageContent] = Field(..., description="The translated content for each page")



@CrewBase
class StoryBookCrew():
    """Story book creation crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            memory=False,
            llm=llm,
        )
    
    @agent
    def story_outline_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['story_outline_planner'],
            verbose=True,
            memory=False,
            llm=llm,
        )
    
    @agent
    def childrens_book_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['childrens_book_writer'],
            verbose=True,
            memory=False,
            llm=llm,
        )

    @agent
    def art_director(self) -> Agent:
        return Agent(
            config=self.agents_config['art_director'],
            verbose=True,
            memory=False,
            llm=llm,
        )

    @agent
    def illustrator(self) -> Agent:
        return Agent(
            config=self.agents_config['illustrator'],
            verbose=True,
            memory=False,
            llm=llm,
            tools=[ImageGenerationTool()]
        )

    @agent
    def translator(self) -> Agent:
        return Agent(
            config=self.agents_config['translator'],
            verbose=True,
            memory=False,
            llm=llm,
        )
    
    @agent
    def page_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['page_designer'],
            verbose=True,
            memory=False,
            llm=llm,
        )

    @task
    def research_story_theme_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_story_theme_task'],
            agent=self.researcher(),
            output_json=ResearchResult
        )

    @task
    def develop_story_outline_task(self) -> Task:
        return Task(
            config=self.tasks_config['develop_story_outline_task'],
            agent=self.story_outline_planner(),
            context=[self.research_story_theme_task()],
            output_json=StoryOutline
        )

    @task
    def write_story_content_task(self) -> Task:
        return Task(
            config=self.tasks_config['write_story_content_task'],
            agent=self.childrens_book_writer(),
            context=[self.develop_story_outline_task()],
            output_json=PageContents
        )

    @task
    def design_art_direction_task(self) -> Task:
        return Task(
            config=self.tasks_config['design_art_direction_task'],
            agent=self.art_director(),
            context=[self.write_story_content_task()],
            output_json=ArtDirection
        )

    @task
    def create_illustrations_task(self) -> Task:
        return Task(
            config=self.tasks_config['create_illustrations_task'],
            agent=self.illustrator(),
            context=[
                self.develop_story_outline_task(),
                self.write_story_content_task(),
                self.design_art_direction_task()
            ],
            tools=[],
            output_json=IllustrationPrompts,
        )
    
    @task
    def generate_illustrations_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_illustrations_task'],
            agent=self.illustrator(),
            context=[self.create_illustrations_task()],
            output_json=Illustrations,
            tools=[ImageGenerationTool()]
        )

    @task
    def translate_content_task(self) -> Task:
        return Task(
            config=self.tasks_config['translate_content_task'],
            agent=self.translator(),
            context=[self.write_story_content_task()],
            output_json=TranslatedContents
        )
    
    @task
    def generate_html_pages_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_html_pages_task'],
            agent=self.page_designer(),
            context=[
                self.write_story_content_task(),
                self.translate_content_task(),
                self.generate_illustrations_task()
            ],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the StoryBook crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            output_handler=lambda x: [task.output for task in x.tasks if task.output]
        )


if __name__ == "__main__":
    # load environment variables
    load_dotenv()

    crew = StoryBookCrew().crew()
    
    result = crew.kickoff({
        'story_theme': 'The modern cosmology about the universe, its birth, evolution and different types of Celestial bodies',
        'age_range': '1-6',
        'target_language': 'Chinese'
    })

    # output the result to a file
    with open('result.json', 'w') as f:
        json.dump(result.model_dump(), open('result.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

    import pdb
    pdb.set_trace()
    
    # Create output directory if it doesn't exist
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'output_htmls_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the template and data from results
    html_template: str = result.raw
    illustrations: list[str] = result.tasks_output[-3].illustration_paths
    english_pages: list[PageContent] = result.tasks_output[-6].pages
    translated_pages: list[PageContent] = result.tasks_output[-2].pages

    # Save the template
    template_file = os.path.join(output_dir, 'template.html')
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(html_template)

    # For each page, create an HTML file with the content
    for i in range(len(illustrations)):
        page_data = {
            'illustration_path': illustrations[i],
            'english_text': english_pages[i].content,
            'translated_text': translated_pages[i].content,
            'english_highlight_vocabulary_word': english_pages[i].core_vocabulary_word,
            'translated_highlight_vocabulary_word': translated_pages[i].core_vocabulary_word
        }
        
        # Render the template with the page data
        page_html = html_template.format(**page_data)
        
        # Save to a file
        output_file = os.path.join(output_dir, f'page_{i+1}.html')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(page_html)
