from agents import Agent

from structured_outputs import TableOfConcepts, TableOfConceptsGuardrail, FollowUpQuestions, NewHypothesis, \
    ChapterText
from tools import search_web_tool


class TableOfConceptsAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="Table of concepts agent",
            instructions="""
Помогите пользователю создать оглавление для его исследовательской работы.
Напишите ответ в формате json.
В title напишите название исследовательской работы
Для каждой главы напишите chapter_name, chapter_description, need_research.
need_research показывает, нужен ли поиск в интернете для того чтобы написать главу.
Почти для всех глав need_research=True, кроме как для тех которые пишутся уже после основного текста исследования.
Например, для заключения и введения need_research=False
            """,
            output_type=TableOfConcepts,
            *args, **kwargs
        )


class TableOfConceptsSearchAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="Table of concepts search agent",
            instructions="""
Помогите пользователю создать оглавление для его исследовательской работы.
Воспользуйся поиском в интернете для составления более релевантного оглавления.
            """,
            tools=[search_web_tool],
            *args, **kwargs
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
1. Вопросы должны быть четкими, конкретными без необходимости дополнительного контекста (текста работы) для их понимания.
3. Включи вопросы, которые проверяют:  
   - Соответствие цели и задач исследования.  
   - Логическую связность аргументации.  
   - Достаточность доказательств и примеров.  
   - Возможные противоречия или альтернативные точки зрения.  
   
Хорошие ворпосы:
Каковы перспективы развития направления supervised-learning?
В чем заключается идея создания СССР?
Каков принцип работы RAG?
Что такое Mixture of Experts?
Какие виды заболеваний не лечатся без хирургического вмешательства?
Что делать человеку в условиях низкого сахара в крови, если он один в лесу?

Плохие вопросы:
Какие смыслы хочет раскрыть автор в ходе иследования?
Какие принципы случше расскрыть в заключеннии текста?
Как исследование должно раскрывать суть этой главы?

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
На основе мыслей и гипотез которые у тебя есть сгенерируй список гипотез которые
можно логически вывести из мыслей которые у тебя есть и уже созданных тобой гипотез. 

Гипотезы должны быть написаны в формате утверждений, а не вопросов.

**Входные данные:**  
Мысли: 
{n.join(context['summaries']) if len(context['summaries']) > 0 else "Пока мыслей нет"}



Гипотезы: 
{n.join(context['hypos']) if len(context['hypos']) > 0 else "Пока гипотез нет"}

Прежде чем ответить хорошо подумай шаг за шагом, воспроизводя цепочку рассужедний.
Если в ходе рассуждений твои гипотезы повторяют входыне данные и ты не придумал ничего нового, то оставть список гипотез пустым. 
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
Текст главы должен расскрывать СУТЬ, обозначенную в названии главы.
Если хочешь написать подводку (введение) для главы, то просто пиши ее и НЕ добавляй заголовок "Введение".
Если хочешь обощить написанные мысли в конце главы и сделать выводы, то просто сделай это и НЕ добавляй заголовок "Заключение".
Разрешается использовать только имеющиеся наработки коллег и их гипотезы.

**Входные данные:**

УЖЕ НАПИСАННАЯ ЧАСТЬ РАБОТЫ, КОТОРУЮ НЕ НУЖНО ПОВТОРЯТЬ.
{context['done_work']}



НАРАБОТКИ КОЛЛЕГ: 
{n.join(context['summaries']) if len(context['summaries']) > 0 else "Пока мыслей нет"}



ГИПОТЕЗЫ КОЛЛЕГ: 
{n.join(context['hypos']) if len(context['hypos']) > 0 else "Пока гипотез нет"}


ДЛЯ НАПИСАНИЯ ГЛАВЫ ИСПОЛЬЗУЙ ФОРМАТ MARKDOWN. 
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
Пиши только текст главы!
Текст главы должен быть расскрывать СУТЬ, обозначенную в названии главы.
Если хочешь написать подводку (введение) для главы, то просто пиши ее и НЕ добавляй заголовок "Введение".
Если хочешь обощить написанные мысли в конце главы и сделать выводы, то просто сделай это и НЕ добавляй заголовок "Заключение".
Эта глава не требует исследований, а должна базироваться главах, которые содержат основные мысли исследования.
Проанализируй основные главы и на базе них напиши эту главу.

Написанные главы:  

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
