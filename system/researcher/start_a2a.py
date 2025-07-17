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
            id='123dsaae',
            name='Write research',
            description='Write research with internet search',
            tags=['research'],
            examples=[
                'Write research for topic The impact of digitalization on the labor market.',
                'Write research for topic Development of the sharing economy in megacities.',
                'Write research for topic The role of small and medium enterprises in the context of economic instability.',
                'Write research for topic The impact of social networks on the formation of public opinion.',
                'Write research for topic Changes in the structure of the family in the 21st century.',
                'Write research for topic Problems of social adaptation of migrants.',
                'Write research for topic The evolution of international relations in the era of globalization.',
                'Write research for topic The impact of populist movements on democratic processes.',
                'Write research for topic Analysis of the impact of sanctions on international politics.',
                'Write research for topic Prospects for the development of artificial intelligence and its impact on society.',
                'Write research for topic The future of quantum computing and its applications.',
                'Write research for topic Data security in the era of the Internet of Things.'
            ],
        )
        my_agent_executor = MyAgentExecutor()
        agent_card = AgentCard(
            name='Research Agent',
            # description='Этот агент поможет создать виртуальную машину и подскажет какие конфигурации можно использовать',
            description='This agent will write research for you about your theme',
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
