from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List


class CompositeActivityItem(BaseModel):
    title: str = Field(description="title of the activity")
    description: str = Field(description="description of the activity. Use at least 25 words")


composite_activity_output_parser = PydanticOutputParser(pydantic_object=CompositeActivityItem)

PROMPTS = {
    "baseline_prompt": PromptTemplate(
        template="""
You are a recommender system of {domain} related activities suitable for young adults.
{format_instructions}

Generate activities.
Generate at most {num_of_activities} activities.
If you know only one activity, which is related to art - generate only one.

Do not generate similar activities. All generated activies should be unique.
        """,
        input_variables=['domain', 'num_of_activities'],
        partial_variables={'format_instructions': composite_activity_output_parser.get_format_instructions()}
    ),
}