import datetime
import json
import os
from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from jinja2 import Template
from elevenlabs.client import ElevenLabs


from v0.pydantic_models import ArtDirection, IllustrationPrompts, Illustrations, PageContent, PageContents, ResearchResult, StoryOutline, TranslatedContents
from v0.tools.image_generation import BatchImageGenerationTool


load_dotenv()


claude = LLM(model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0", temperature=0.8)
claude_low_tmp = LLM(model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0", temperature=0.1)
deepseek_r1 = LLM(model="deepseek/deepseek-reasoner", temperature=0.8)  # TODO 框架不支持
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
eleven_labs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)



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
            llm=claude,
        )
    
    @agent
    def story_outline_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['story_outline_planner'],
            verbose=True,
            memory=False,
            llm=claude,
        )
    
    @agent
    def childrens_book_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['childrens_book_writer'],
            verbose=True,
            memory=False,
            llm=claude,
        )

    @agent
    def art_director(self) -> Agent:
        return Agent(
            config=self.agents_config['art_director'],
            verbose=True,
            memory=False,
            llm=claude,
        )

    @agent
    def illustrator(self) -> Agent:
        return Agent(
            config=self.agents_config['illustrator'],
            verbose=True,
            memory=False,
            llm=claude_low_tmp,
            tools=[BatchImageGenerationTool()]
        )

    @agent
    def translator(self) -> Agent:
        return Agent(
            config=self.agents_config['translator'],
            verbose=True,
            memory=False,
            llm=claude_low_tmp,
        )
    
    @agent
    def page_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['page_designer'],
            verbose=True,
            memory=False,
            llm=claude,
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
            context=[self.develop_story_outline_task(), self.write_story_content_task()],
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
            tools=[BatchImageGenerationTool()]
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
    

def generate_audio(text: str, output_dir: str, file_name: str) -> str:
    response = eleven_labs_client.text_to_speech.convert(
        text=text,
        voice_id="XfNU2rGpBa01ckF309OY",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_192",
    )
    # save to file
    output_file = os.path.join(output_dir, file_name)
    with open(output_file, 'wb') as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    return output_file

def generate_html_pages(result: dict, output_dir: str) -> None:
    """
    Generate HTML pages from the crew result.
    
    Args:
        result: The result dictionary from crew.kickoff()
        output_dir: Directory to save the generated HTML files
    """
    # Get the template and data from results
    html_template: str = result['raw']

    # Remove markdown code block if present
    if html_template.startswith('```html'):
        html_template = html_template[7:]
    if html_template.endswith('```'):
        html_template = html_template[:-3]

    html_template = html_template.strip()

    illustrations: list[str] = result['tasks_output'][-3]['json_dict']['illustration_paths']
    english_pages: list[PageContent] = result['tasks_output'][-6]['json_dict']['pages']
    translated_pages: list[PageContent] = result['tasks_output'][-2]['json_dict']['pages']


    english_pages_text = '\n'.join([page['content'] for page in english_pages])
    # for eleven labs
    audio_file_name = 'audio.mp3'
    generate_audio(english_pages_text, output_dir, audio_file_name)

    # Save the template
    template_file = os.path.join(output_dir, 'template.html')
    with open(template_file, 'w+', encoding='utf-8') as f:
        f.write(html_template)
    
    page_htmls = []
    # For each page, create an HTML file with the content
    for i in range(len(illustrations)):
        page_data = {
            'illustration_path': illustrations[i],
            'english_text': english_pages[i]['content'],
            'translated_text': translated_pages[i]['content'],
            'english_highlight_vocabulary_word': english_pages[i]['core_vocabulary_word'],
            'translated_highlight_vocabulary_word': translated_pages[i]['core_vocabulary_word']
        }
        
        # Render the template with the page data
        # Only format the keys that exist in page_data
        page_html = html_template
        
        # Create template object and render with page data
        template = Template(page_html)
        page_html = template.render(**page_data)
        
        # Save to a file
        output_file = os.path.join(output_dir, f'page_{i+1}.html')
        with open(output_file, 'w+', encoding='utf-8') as f:
            f.write(page_html)
        page_htmls.append(page_html)

    # Create a merged HTML file combining all pages horizontally
    merged_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            padding: 20px;
            display: flex;
            overflow-x: auto;
            min-height: 100vh;
        }}
        .page-container {{
            display: flex;
            gap: 20px;
        }}
        .page {{
            flex: 0 0 auto;
            width: 21cm; /* A4 width */
            height: 29.7cm; /* A4 height */
            margin-right: 20px;
        }}
        iframe {{
            width: 100%;
            height: 120%;
            border: 1px solid #ccc;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        #audio-player {{
            position: fixed;
            top: 20px; /* Hide below viewport */
            right: 20px;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <audio id="audio-player" controls autoplay>
        <source src="{audio_file_name}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    <div class="page-container">
        {"".join([f'<div class="page"><iframe src="page_{i+1}.html" frameborder="0"></iframe></div>' for i in range(len(page_htmls))])}
    </div>
</body>
</html>
"""

    # Save the merged file
    merged_file = os.path.join(output_dir, 'merged_book.html')
    with open(merged_file, 'w+', encoding='utf-8') as f:
        f.write(merged_html)


def generate_story_book(story_theme: str, age_range: str, target_language: str, output_dir: str | None = None) -> dict:
    """
    Generate a complete story book with illustrations and translations.
    
    Args:
        story_theme: Theme/topic of the story
        age_range: Target age range (e.g. '1-6')
        target_language: Language to translate the story into
        output_dir: Optional directory to save HTML output files. If None, uses timestamped directory.
        
    Returns:
        dict: The complete result dictionary containing all story content
    """
    load_dotenv()

    crew = StoryBookCrew().crew()
    
    result = crew.kickoff({
        'story_theme': story_theme,
        'age_range': age_range, 
        'target_language': target_language
    })

    # Generate HTML pages if output_dir specified
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'output_htmls_{timestamp}'

    # Convert to dict and save result
    result_dict = result.model_dump()
    os.makedirs(output_dir, exist_ok=True)
    result_json_path = os.path.join(output_dir, 'result.json')
    with open(result_json_path, 'w+', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
    
    generate_html_pages(result_dict, output_dir)
    
    return result_dict


if __name__ == "__main__":
    # The modern cosmology about the universe, its birth, evolution and different types of Celestial bodies
    # A Brief History of Time: from the Big Bang to Black Holes
    generate_story_book(
        story_theme='Little firefighter James put out the fire and saved everyone',
        age_range='1-6',
        target_language='Chinese'
    )
    # generate_audio("test it!", '', 'test.mp3')