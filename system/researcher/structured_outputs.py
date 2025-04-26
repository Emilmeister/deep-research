from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RelevanceScoreNumber(int, Enum):
    """Оценка релевантности  (0-10)."""
    RELEVANCE_0 = 0
    RELEVANCE_1 = 1
    RELEVANCE_2 = 2
    RELEVANCE_3 = 3
    RELEVANCE_4 = 4
    RELEVANCE_5 = 5
    RELEVANCE_6 = 6
    RELEVANCE_7 = 7
    RELEVANCE_8 = 8
    RELEVANCE_9 = 9
    RELEVANCE_10 = 10


class Chapter(BaseModel):
    chapter_name: str
    chapter_description: str
    need_research: bool


class TableOfConcepts(BaseModel):
    title: str
    chapters: list[Chapter]

    def print(self):
        chapters = "\n".join([f"## {chapter.chapter_name}\n{chapter.chapter_description}" for chapter in self.chapters])
        return f"# {self.title}\n{chapters}"


class TableOfConceptsGuardrail(BaseModel):
    explanation: str
    satisfied: bool


class FollowUpQuestions(BaseModel):
    questions: list[str]


class InterestingUrl(BaseModel):
    web_page_url: str
    why_interested: str
    question_and_url_relevant_score: RelevanceScoreNumber


class SummaryWithInterestingUrls(BaseModel):
    summary: str
    relevance_score: RelevanceScoreNumber
    interesting_web_page_urls: list[InterestingUrl]


class RelevanceScore(BaseModel):
    reasoning: str
    relevance_score: RelevanceScoreNumber


class ChapterText(BaseModel):
    chapter_title: str
    chapter_text_without_title_in_head: str


class NewHypothesis(BaseModel):
    reasoning: str
    list_of_brilliant_ideas: list[str]


class NeedRewriteTableOfConcepts(BaseModel):
    need_rewrite: bool
    explanation: str
    new_table_of_concepts: Optional[TableOfConcepts]


class SearchWords(BaseModel):
    words: list[str]
