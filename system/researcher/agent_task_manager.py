from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Task,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError
from a2a_agent import ResearchAgent


class MyAgentExecutor(AgentExecutor):
    """AgentExecutor Example."""

    def __init__(self):
        self.agent = ResearchAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task

        # This agent always produces Task objects. If this request does
        # not have current task, create a new one and use it.
        if not task:
            task = new_task(context.message)
            event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.contextId)
        # invoke the underlying agent, using streaming results. The streams
        # now are update events.
        async for item in self.agent.stream(query, task.contextId):
            is_task_complete = item['is_task_complete']
            require_user_input = item['require_user_input']
            is_error = item['is_error']

            if is_error:
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(
                        item['content'], task.contextId, task.id
                    ),
                )
                break
            if not is_task_complete and not require_user_input:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        item['content'], task.contextId, task.id
                    ),
                )
                continue

            if not is_task_complete and require_user_input:
                await updater.update_status(
                    TaskState.input_required,
                    new_agent_text_message(
                        item['content'], task.contextId, task.id
                    ),
                )
                break
            if is_task_complete and not require_user_input:
                await updater.update_status(
                    TaskState.completed,
                    new_agent_text_message(
                        item['content'], task.contextId, task.id
                    ),
                )
                break

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
