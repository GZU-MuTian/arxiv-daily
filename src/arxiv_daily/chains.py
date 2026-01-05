from . import llm_client

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser

from pydantic import BaseModel, Field
from rich.console import Console
from rich.text import Text
import logging

logger = logging.getLogger(__name__)
console = Console()


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

# --- ConceptExtractorChain ---

class ConceptExtractor(BaseModel):
    concept: str = Field(..., description="The extracted concept phrase.")
    concept_class: str = Field(..., alias="class", description="The category of the concept.")
    description: str = Field(..., description="Technical description (~100 words)")

    def __rich__(self) -> Text:
        text = Text()
        for k, v in self.model_dump().items():
            name = k.replace("_", " ").title()
            text.append(f"{name}: ", style="bold white")
            text.append(f"{v}\n", style="white")
        return text

_SYSTEM_EXTRACTOR_PROMPT = """
You are an AI specializing in astrophysics, tasked with extracting key concepts from journal articles and providing technical descriptions for each. These concepts will be used to construct a knowledge graph, so focus on identifying the most relevant and informative concepts. Emphasize key innovations of the papers rather than references in the introduction. Extract both scientific concepts in astronomy and technological concepts, including techniques in machine learning, statistics, and numerical simulations.

Consider the following guidelines when extracting concepts:

**Relevance**: Ensure the concepts are directly related to the core findings and innovations of the paper.

**Clarity**: Extract clear and specific concepts that can be easily understood and categorized. Aim to limit each concept to three to four words.

**Classification**: Identify the appropriate class for each concept. Classes can include:

- Galaxy Physics (e.g., "Galaxy Formation", "Spiral Galaxies", "Dwarf Galaxies", "Intergalactic Medium", "Galactic Nuclei")

- Cosmology & Nongalactic Physics (e.g., "Dark Matter", "Cosmic Microwave Background", "Large-Scale Structure", "Cosmic Inflation", "Cosmological Parameters")

- Earth & Planetary Science (e.g., "Exoplanet Detection", "Planetary Atmospheres", "Astrobiology", "Planetary Formation", "Solar System Evolution")

- High Energy Astrophysics (e.g., "Black Hole Physics", "Neutron Stars", "Gamma-Ray Bursts", "Supernovae", "High-Energy Cosmic Rays")

- Solar & Stellar Physics (e.g., "Stellar Evolution", "Solar Flares", "Star Formation", "Stellar Atmospheres", "Helioseismology")

- Statistics & AI (e.g., "Machine Learning Algorithms", "Bayesian Inference", "Neural Networks", "Statistical Analysis", "Data Mining")

- Numerical Simulation (e.g., "N-body Simulations", "Hydrodynamic Simulations", "Radiative Transfer", "Simulation Codes", "Computational Astrophysics")

- Instrumental Design (e.g., "Telescope Design", "Spectrographs", "Detector Technology", "Observational Techniques", "Space Telescopes")

Note that the examples above are just purely for reference, unless they fit as concepts in the paper, do not use them. Aim to extract about 10 key concepts for each paper.

For each concept, also provide a concise technical description (~100 words) explaining its general principles and significance in astronomy. While you may reference the paper's context, focus on broadly defining and explaining each concept.

**Description Guidelines:**
1. Ensure each description is technically precise and suitable for an astronomy expert audience
2. Do not use backslashes or special characters in your descriptions as they can cause JSON parsing errors
3. Use only verbal descriptions - no mathematical equations or formulas
4. Focus on clear, conceptual explanations using words
5. Keep concept names exactly as provided - do not add plurals, capitalization, or prepositions
6. Don't output any RA and Dec information in the description

Provide your response as a JSON array of concept objects. Begin your output with the JSON structure immediately, without any preceding text. Strictly adhere to the specified output format.
Output format (JSON array):
[
    {{
        "concept": "The extracted concept phrase.",
        "class": "The category of the concept.",
        "description": "Technical description (~100 words)"
    }}
]
"""

_HUMAN_EXTRACTOR_PROMPT = """
Summary: {organized_summary}

Extract approximately 10 key concepts. Output only valid JSON array, no other text.
"""

EXTRACTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_EXTRACTOR_PROMPT),
    ("human", _HUMAN_EXTRACTOR_PROMPT)
])

def ConceptExtractorChain():
    """
    Chain that extracts astronomical concepts from a paper summary.
    """
    llm = llm_client.getLLM()
    logger.info("LLM configuration: {}".format(llm.model_dump(exclude_unset=True)))
    return EXTRACTOR_PROMPT | llm | JsonOutputParser()
