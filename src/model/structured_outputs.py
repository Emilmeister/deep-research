from typing import Optional

from pydantic import BaseModel


class Chapter(BaseModel):
    chapter_name: str
    chapter_description: str
    need_research: bool


class TableOfConcepts(BaseModel):
    title: str
    chapters: list[Chapter]


class FollowUpQuestions(BaseModel):
    questions: list[str]


class SummaryWithInterestingUrls(BaseModel):
    summary: str
    interesting_urls: list[str]


class NeedRewriteTableOfConcepts(BaseModel):
    need_rewrite: bool
    explanation: str
    new_table_of_concepts: Optional[TableOfConcepts]


