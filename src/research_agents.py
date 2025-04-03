from agents.models import openai_provider
from openai import AsyncOpenAI
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, set_default_openai_client, \
    set_default_openai_api, set_trace_processors
from pydantic import BaseModel
import asyncio

from src.model.structured_outputs import TableOfConcepts, TableOfConceptsGuardrail, FollowUpQuestions, NewHypothesis, \
    ChapterText


class TableOfConceptsAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Table of concepts agent",
            instructions="""
            Помогите пользователю создать оглавление для его исследовательской работы.
            Напишите ответ в формате json.
            Для каждой главы напишите chapter_name, chapter_description, need_research.
            Например, для заключения и введения исследование need_research=False
            """,
            output_type=TableOfConcepts
        )


class TableOfConceptsGuardrailAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Table of concepts guardrail agent",
            instructions="""
            Не отвечайте на сообщение пользователя.
            Вы агент, который только проверяет, удовлетворен ли пользователь оглавлением или нет.
            Проанализируйте его сообщения, и если под конец диалога его что то не устраиватет, 
            и он хочет что поменять в оглавлении, то значит он не удовлетворен.
            Если же он говорит, что все хорошо, то он удовлетворен.
            """,
            output_type=TableOfConceptsGuardrail
        )


def follow_up_questions_agent_sys_prompt(context, agent):
    n = '\n'
    context = context.context
    return f"""
            На основе названия и описания главы исследовательской работы сгенерируй список релевантных и содержательных вопросов для поиска в интернете, 
            которые помогут глубже раскрыть тему, проверить логику изложения и выявить потенциальные пробелы. 

            **Входные данные:**  
            Название главы: {context['chapter_name']}  
            Описание главы: {context['chapter_description']}
            Сырые мысли: 
            {n.join(context['summaries']) if len(context['summaries']) > 0 else "Пока мыслей нет"}
            
            
            
            Гипотезы: 
            {n.join(context['hypos']) if len(context['hypos']) > 0 else "Пока гипотез нет"}
            
            
            
            **Требования к вопросам:**  
            1. Вопросы должны быть четкими, конкретными без необходимости дополнительного контекста для их понимания.
            3. Включи вопросы, которые проверяют:  
               - Соответствие цели и задач исследования.  
               - Логическую связность аргументации.  
               - Достаточность доказательств и примеров.  
               - Возможные противоречия или альтернативные точки зрения.  
            
            **Формат вывода:**  
            - Список вопросов в логическом порядке, соответствующий структуре главы.  
            - Каждый вопрос должен быть сформулирован в нейтрально-аналитическом тоне. 
            """


class FollowUpQuestionsAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Follow up questions agent",
            instructions=follow_up_questions_agent_sys_prompt,
            output_type=FollowUpQuestions
        )


def hypos_generating_agent_sys_prompt(context, agent):
    n = '\n'
    context = context.context
    return f"""
            На основе мыслей и гипотез которые у тебя есть сгенерируй список новых идей и гипотез которые
            можно логически вывести из мыслей которые у тебя есть и уже созданных тобой гипотез. 

            **Входные данные:**  
            Сырые мысли: 
            {n.join(context['summaries']) if len(context['summaries']) > 0 else "Пока мыслей нет"}
            
            
            
            Гипотезы: 
            {n.join(context['hypos']) if len(context['hypos']) > 0 else "Пока гипотез нет"}
            
            Прежде чем ответить хорошо подумай шаг за шагом, воспроизводя цепочку рассужедний.
            """


class HyposGeneratingAgent(Agent):
    def __init__(self):
        super().__init__(
            name="hypos generation agent",
            instructions=hypos_generating_agent_sys_prompt,
            output_type=NewHypothesis
        )


def chapter_editor_agent_sys_prompt(context, agent):
    n = '\n'
    context = context.context
    return f"""
            Ты редактор статей для научных журналов.
            Твоя задача написать одну единственную главу под названием \"{context['chapter_name']}\" 
            для исследования на тему \"{context['title']}\".
            Не вставляй в начало название главы. Другой редактор добавит ее за тебя самостоятельно!
            Не пиши исследование целиком, а только главу!
            Разрешается использовать только имеющиеся наработки коллег и их гипотезы.

            **Входные данные:**  
            
            Наработки коллег: 
            {n.join(context['summaries']) if len(context['summaries']) > 0 else "Пока мыслей нет"}
            
            
            
            Гипотезы коллег: 
            {n.join(context['hypos']) if len(context['hypos']) > 0 else "Пока гипотез нет"}
            
            
            Для написания главы используй формат markdown. 
            """


class ChapterEditorAgent(Agent):
    def __init__(self):
        super().__init__(
            name="chapter generation agent",
            instructions=chapter_editor_agent_sys_prompt,
            output_type=ChapterText
        )


def chapter_editor_summary_agent_sys_prompt(context, agent):
    n = '\n\n'
    context = context.context

    chapters = [f"# Название главы: {chapter_name}\n\n# Текст главы:\n {chapter_text}" for chapter_name, chapter_text in context['done_chapters'].items()]
    return f"""
            Ты редактор статей для научных журналов.
            Твоя задача написать одну единственную главу под названием \"{context['chapter_name']}\" 
            для исследования на тему \"{context['title']}\".
            Не вставляй в начало название главы. Другой редактор добавит ее за тебя самостоятельно!
            Не пиши исследование целиком, а только главу!
            Не делай в конце общее заключее, а только промежуточный итог, так как общее заключение по всей работе будет делать другой редактор в другой главе!
            Эта глава не требует исследований, а должна базироваться главах, которые содержат основные мысли исследования.
            Проанализируй основные главы и на базе них напиши эту главу.

            # Написанные главы:  

            {n.join(chapters)}
            
            
            Для написания используй формат markdown. 
            """


class ChapterEditorSummaryAgent(Agent):
    def __init__(self):
        super().__init__(
            name="chapter generation agent",
            instructions=chapter_editor_summary_agent_sys_prompt,
            output_type=ChapterText
        )
