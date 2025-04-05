from typing import Optional

from pydantic import BaseModel, Field


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
    question_and_url_relevant_score: int = Field(..., ge=0, le=10)


class SummaryWithInterestingUrls(BaseModel):
    summary: str
    relevance_score: int = Field(..., ge=0, le=10)
    interesting_web_page_urls: list[InterestingUrl] = []


class RelevanceScore(BaseModel):
    reasoning: str
    relevance_score: int = Field(..., ge=0, le=10)


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
