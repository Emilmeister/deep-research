import logging
import os

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

from agent_task_manager import MyAgentExecutor



load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""

    pass


def main():
    try:
        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id='any_vm_answer',
            name='Создать виртуальную машину',
            description='Создать виртуальную машину пользователю',
            tags=['виртуальная машина', 'создать'],
            examples=[
                'Запусти виртуальную машину',
            ],
        )
        my_agent_executor = MyAgentExecutor()
        agent_card = AgentCard(
            name='VM Creator',
            # description='Этот агент поможет создать виртуальную машину и подскажет какие конфигурации можно использовать',
            description='This agent will help you create a virtual machine and tell you what configurations can be used',
            url=f'http://{os.getenv("A2A_HOST")}:{os.getenv("A2A_PORT")}/',
            version='1.0.0',
            defaultInputModes=my_agent_executor.agent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=my_agent_executor.agent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        request_handler = DefaultRequestHandler(
            agent_executor=my_agent_executor,
            task_store=InMemoryTaskStore(),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )
        import uvicorn

        uvicorn.run(server.build(), host=os.getenv("A2A_HOST"), port=int(os.getenv("A2A_PORT")))
    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()
