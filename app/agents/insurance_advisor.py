from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.agents.message_history import ToolCallHistoryMiddleware
from app.agents.tools import query_candidate_products,save_insurance_plan,query_product_clause
from app.core.config import settings
from app.core.logging import get_logger
from app.agents.schemas import InsuranceAgentContext
logger = get_logger(__name__)


SYSTEM_PROMPT = """
你是“安心保”的智能保险顾问。
- 你需要使用专业、准确、容易理解的语言回答用户的保险问题。
- 当信息不足时，应先向用户追问，形成用户画像。
- 给用户推荐保险方案时，必须先查询候选产品，根据用户画像，推荐合适的保险组合，再调用工具保存方案。
- 回答保险产品的条款问题时，必须在当前这一轮先调用 query_product_clause 查询对应产品的条款，再根据本次返回的片段回答。不要复用上一轮的检索结果或引用编号。
- 每个事实陈述必须紧跟本次返回里真正包含该事实的 [source_id]。例如：这款产品的等待期为180天。[ref-001]
- 不同片段支持的事实要用各自的编号，不要把所有事实都标成 [ref-001]。本次没有查到条款时，不要编造引用编号。
- 回答要简短、精确，不要长篇大论。
"""


async def init_insurance_advisor():
    """初始化保险顾问Agent"""

    # 1.初始化模型
    model = init_chat_model(
        model=settings.llm.chat_model,
        model_provider="deepseek",
        api_key=settings.llm.api_key,
        extra_body={"thinking": {"type": "disabled"}},
    )

    # 2.创建Agent 
    agent = create_agent(
        model=model,
        tools=[query_candidate_products,save_insurance_plan,query_product_clause],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=True,
        context_schema=InsuranceAgentContext,    # Context约束
        middleware=[ToolCallHistoryMiddleware()],
    )
    logger.info("保险顾问Agent初始化成功~✅️")
    return agent