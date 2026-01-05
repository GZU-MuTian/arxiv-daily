"""
Algorithm inspired by: arXiv:2511.12353.
"""
from . import llm_client

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


# --- PaperCompressionChain ---

_SYSTEM_COMPRESS_PROMPT = """
You are an AI specializing in astrophysics, tasked with condensing astrophysics journal texts. Adhere to these guidelines:
1. Retain LaTeX code for formulas, remove other LaTeX symbols.
2. Exclude acknowledgments and appendices at the end of the paper.
3. Emphasize the paper's motivations, novel technical details, key theories, and concepts.
4. Highlight innovative results and their links to other works.
5. Integrate information from figures' captions, omit figures.
6. Clarify or maintain technical jargon at the level that is clear for astrophysics researchers.
7. Convey the author's perspective and interpretation of results.
Consider context from previous parts when summarizing individual sections. Exclude references at the end. Current context:\n\n{context}
"""

_HUMAN_COMPRESS_PROMPT = """
Condense the following text into a maximum of {num_word_limit} words, avoiding repetition of the provided context. Exclude references at the end. Paragraphs:\n\n{chunk}
"""

COMPRESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_COMPRESS_PROMPT),
    ("human", _HUMAN_COMPRESS_PROMPT)
])

def PaperCompressionChain():
    llm = llm_client.getLLM()
    logger.info("LLM configuration: {}".format(llm.model_dump(exclude_unset=True)))
    return COMPRESS_PROMPT | llm | StrOutputParser()

# --- OrganizedSummaryChain ---

class OrganizedSummary(BaseModel):
    background: str = Field("", description="Scientific context and prior work.")
    motivation: str = Field("", description="Why this study is needed.")
    methodology: str = Field("", description="Approach, models, or simulations used.")
    results: str = Field("", description="Key findings or outcomes.")
    interpretation: str = Field("", description="What the results mean.")
    implication: str = Field("", description="Broader impact or future directions.")

organized_summary_parser = PydanticOutputParser(pydantic_object=OrganizedSummary)

_SYSTEM_ORGANIZER_PROMPT = """
You are an AI specializing in astrophysics, tasked with reorganizing astrophysics paper summaries. 
Adhere to these guidelines:

1. Reorganize the summary strictly into the following key areas and nothing else:
   - Background
   - Motivation
   - Methodology
   - Results
   - Interpretation
   - Implication

2. **Writing style:**
   - Use THIRD PERSON only: "this study", "the authors", "the paper"
   - NEVER use first person: no "we", "our", "I"
   - Remove ALL references to specific sections, figures, tables, or appendices (e.g., "Section 3", "Figure 2", "Table 1")
   - Write in continuous narrative form, avoiding bullet points or lists

3. **Logical flow:**
   - Background → Motivation should connect naturally (Background sets context, Motivation explains why this work matters)
   - Motivation → Methodology should flow logically (Motivation identifies gap, Methodology describes approach)
   - Methodology → Results should be connected (Methods used lead to Results obtained)
   - Results → Interpretation → Implication should build on each other progressively

4. Ensure as much as possible information from the original summary is included.
5. Do not add any new information beyond what is already in the summary.
6. Retain any LaTeX formulas present in the original summary.
7. Keep technical jargon intact, as it's meant for astrophysics researchers.
8. Ensure the output is valid JSON that can be parsed. Be careful with escape characters - use proper JSON escaping for quotes, backslashes, and other special characters.\n\n{format_instructions}
"""

_HUMAN_ORGANIZER_PROMPT = """
Please reorganize the following astrophysics paper summary strictly into the key areas specified in the guidelines, outputting as valid JSON. Here's the summary:\n\n{summary}
"""

ORGANIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_ORGANIZER_PROMPT),
    ("human", _HUMAN_ORGANIZER_PROMPT)
]).partial(format_instructions=organized_summary_parser.get_format_instructions())

def OrganizedSummaryChain():
    llm = llm_client.getLLM()
    logger.info("LLM configuration: {}".format(llm.model_dump(exclude_unset=True)))
    return ORGANIZER_PROMPT | llm | organized_summary_parser
