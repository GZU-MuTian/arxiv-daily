"""
Algorithm inspired by: arXiv:2511.12353.
"""
from . import llm_client

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List
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

# --- KnowledgeGraphExtractor ---

class Relationship(BaseModel):
    concept: str = Field(..., description="Concept name (3-4 words, singular form, lowercase, words joined by hyphens).")
    category: str = Field(..., description="Category class (e.g., galaxy-physics, cosmology, statistics-ai, numerical-simulation, instrumental-design, astronomical-events).")
    relation: str = Field(..., description="Predicate describing what this paper does with the concept (e.g., detects, constrains, simulates, analyzes, discovers).")
    description: str = Field(..., description="1-2 sentence summary describing how this relationship manifests in the paper, written in your own words.")

class KnowledgeGraphExtraction(BaseModel):
    relationships: List[Relationship] = Field(default_factory=list, description="List of relationships from this paper to concepts, with category, relation, and description.")

knowledge_graph_parser = PydanticOutputParser(pydantic_object=KnowledgeGraphExtraction)

_SYSTEM_KG_PROMPT = """
You are an AI specializing in astrophysics and knowledge graph construction. Your task is to extract key concepts and their relationships from academic papers to build a structured knowledge graph.

Adhere to these guidelines:

1. **Concept & Relationship Extraction**:
   - Extract ~10 key concepts that represent the core innovations and contributions of the paper
   - For each concept, define a relationship describing what this paper does with it
   - The subject is always this paper (identified by its arXiv ID); do NOT extract it separately
   - Focus on scientific concepts in astronomy/astrophysics AND technological concepts (ML, statistics, simulations, instrumentation)
   - Limit concept names to 3-4 words, use singular form, lowercase
   - Avoid concepts from introduction/background that are merely references to prior work

2. **Category Classification** — assign each concept to one of these classes:
   - galaxy-physics (e.g., galaxy-formation, spiral-galaxies, intergalactic-medium)
   - cosmology (e.g., dark-matter, cosmic-microwave-background, large-scale-structure)
   - earth-planetary (e.g., exoplanet-detection, planetary-atmospheres, astrobiology)
   - high-energy-astrophysics (e.g., black-hole-physics, neutron-stars, gamma-ray-bursts)
   - solar-stellar (e.g., stellar-evolution, solar-flares, star-formation)
   - statistics-ai (e.g., machine-learning, bayesian-inference, neural-networks)
   - numerical-simulation (e.g., n-body-simulations, hydrodynamic-simulations, radiative-transfer)
   - instrumental-design (e.g., telescope-design, spectrographs, detector-technology)
   - astronomical-events (e.g., gravitational-wave-event, fast-radio-burst, tidal-disruption-event, supernova, nova)

   **Important**: Specific astronomical event designations must be extracted as individual concept instances under this class. Do not treat events merely as generic strings — they are important nodes in the knowledge graph (e.g., extract "GW170817" or "GRB 170817A" as distinct concepts).

3. **Relationship Attributes**:
   - `relation`: What this paper does with the concept (e.g., introduces, utilizes, detects, constrains, simulates, analyzes, discovers, validates, measures)
   - `description`: 1-2 sentence summary describing how this relationship manifests in the paper, written in your own words (not a direct quote)
   - Extract 3-8 meaningful relationships with clear descriptions

4. **Output Format**:
   - Return valid JSON that can be parsed
   - Use proper JSON escaping for quotes, backslashes, and special characters
   - Begin output immediately with JSON structure, no preceding text

{format_instructions}
"""

_HUMAN_KG_PROMPT = """
Extract key concepts and relationships from the following paper summary. Here's the text:\n\n{summary}
"""

KNOWLEDGE_GRAPH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_KG_PROMPT),
    ("human", _HUMAN_KG_PROMPT)
]).partial(format_instructions=knowledge_graph_parser.get_format_instructions())

def KnowledgeGraphExtractor():
    llm = llm_client.getLLM()
    logger.info("LLM configuration: {}".format(llm.model_dump(exclude_unset=True)))
    return KNOWLEDGE_GRAPH_PROMPT | llm | knowledge_graph_parser