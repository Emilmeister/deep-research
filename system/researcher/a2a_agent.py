from typing import Dict, Any, List, AsyncGenerator
import asyncio

from main import generate_table_of_concepts, generate_research


class RunnerState:
    history: List = []
    table_of_concepts: dict = None
    research_started: bool = False

    def __init__(self):
        self.history = []
        self.table_of_concepts = None
        self.research_started = False



class ResearchAgent:
    def __init__(self):
        # Initialize runner storage
        self.runners: Dict[str, RunnerState] = {}

    def _get_or_create_runner(self, session_id: str) -> RunnerState:
        """Get an existing runner or create a new one for the session."""
        if session_id not in self.runners:
            self.runners[session_id] = RunnerState()
        return self.runners[session_id]

    async def invoke(self, query: str, session_id: str) -> Dict[str, Any]:
        """Asynchronous invocation of the agent."""
        """Stream the agent's processing and responses."""
        runner_state = self._get_or_create_runner(session_id)

        if not runner_state.table_of_concepts:
            # Complete run
            table_of_concepts = await generate_table_of_concepts(query, runner_state.history)
            runner_state.table_of_concepts = table_of_concepts

            # Format the response
            return {
                "is_task_complete": False,
                "require_user_input": True,
                "content": runner_state.history[-1].content
            }
        else:
            if runner_state.research_started:
                return {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Идет исследование. Ждите."
                }
            # Complete run
            runner_state.research_started = True
            try:
                table_of_concepts = await generate_table_of_concepts(query, runner_state.history)
                runner_state.table_of_concepts = table_of_concepts

                research_chapter_states = []
                for research_chapter_state in generate_research(table_of_concepts, runner_state.history):
                    research_chapter_states.append(research_chapter_state)
                    if research_chapter_state['final']:
                        break

                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": research_chapter_states[-1]['research']
                }
            except Exception as e:
                print(e)
                return {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "is_error": True,
                    "content": "Произошла ошибка при обработке запроса."
                }

    async def stream(self, query: str, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream the agent's processing and responses."""
        runner_state = self._get_or_create_runner(session_id)

        if not runner_state.table_of_concepts:
            try:
                # Complete run
                table_of_concepts = await generate_table_of_concepts(query, runner_state.history)
                runner_state.table_of_concepts = table_of_concepts

                print("runner_state.history generate_table_of_concepts ", runner_state.history)
                # Format the response
                yield {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": table_of_concepts.print(),
                    "is_error": False
                }
            except Exception as e:
                print(e)
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Произошла ошибка при обработке запроса.",
                    "is_error": True
                }
        else:
            if runner_state.research_started:
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Идет исследование. Ждите.",
                    "is_error": False
                }
            # Complete run
            runner_state.research_started = True
            try:
                print("runner_state.history generate_research", runner_state.history)
                table_of_concepts = await generate_table_of_concepts(query, runner_state.history)
                runner_state.table_of_concepts = table_of_concepts

                async for research_chapter_state in generate_research(table_of_concepts, runner_state.history):
                    if research_chapter_state['final']:
                        yield {
                            "is_task_complete": True,
                            "require_user_input": False,
                            "content": research_chapter_state['research'],
                            "is_error": False
                        }
                        break
                    else:
                        yield {
                            "is_task_complete": False,
                            "require_user_input": False,
                            "content": research_chapter_state['progress'],
                            "is_error": False
                        }
            except Exception as e:
                print(e)
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Произошла ошибка при обработке запроса.",
                    "is_error": True
                }

    # For compatibility with the original implementation
    def sync_invoke(self, query: str, session_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for invoke."""
        return asyncio.run(self.invoke(query, session_id))

    # For compatibility with the original API
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]