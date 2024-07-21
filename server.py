import re
from pathlib import Path
from typing import Any, Callable, Dict, Union
from operator import itemgetter

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core import __version__
from langchain import hub
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import ConfigurableFieldSpec, RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict
from supabase import create_client
from retriever import SupabaseRetriever
from langchain_community.chat_models.azureml_endpoint import AzureMLEndpointApiType, AzureMLChatOnlineEndpoint
import cohere
import os


from langserve import add_routes

from dotenv import load_dotenv

load_dotenv()

supabase = create_client(os.getenv('DB_LINK'), os.getenv('DB_KEY'))

co = cohere.Client(
    base_url=os.getenv('COHERE_URL'), api_key=os.getenv('COHERE_KEY')
)


# Define the minimum required version as (0, 1, 0)
# Earlier versions did not allow specifying custom config fields in
# RunnableWithMessageHistory.
MIN_VERSION_LANGCHAIN_CORE = (0, 1, 0)

# Split the version string by "." and convert to integers
LANGCHAIN_CORE_VERSION = tuple(map(int, __version__.split(".")))

if LANGCHAIN_CORE_VERSION < MIN_VERSION_LANGCHAIN_CORE:
    raise RuntimeError(
        f"Minimum required version of langchain-core is {MIN_VERSION_LANGCHAIN_CORE}, "
        f"but found {LANGCHAIN_CORE_VERSION}"
    )


def _is_valid_identifier(value: str) -> bool:
    """Check if the value is a valid identifier."""
    # Use a regular expression to match the allowed characters
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))


def create_session_factory(
    base_dir: Union[str, Path],
) -> Callable[[str], BaseChatMessageHistory]:
    """Create a factory that can retrieve chat histories.

    The chat histories are keyed by user ID and conversation ID.

    Args:
        base_dir: Base directory to use for storing the chat histories.

    Returns:
        A factory that can retrieve chat histories keyed by user ID and conversation ID.
    """
    base_dir_ = Path(base_dir) if isinstance(base_dir, str) else base_dir
    if not base_dir_.exists():
        base_dir_.mkdir(parents=True)

    def get_chat_history(user_id: str, conversation_id: str) -> FileChatMessageHistory:
        """Get a chat history from a user id and conversation id."""
        if not _is_valid_identifier(user_id):
            raise ValueError(
                f"User ID {user_id} is not in a valid format. "
                "User ID must only contain alphanumeric characters, "
                "hyphens, and underscores."
                "Please include a valid cookie in the request headers called 'user-id'."
            )
        if not _is_valid_identifier(conversation_id):
            raise ValueError(
                f"Conversation ID {conversation_id} is not in a valid format. "
                "Conversation ID must only contain alphanumeric characters, "
                "hyphens, and underscores. Please provide a valid conversation id "
                "via config. For example, "
                "chain.invoke(.., {'configurable': {'conversation_id': '123'}})"
            )

        user_dir = base_dir_ / user_id
        if not user_dir.exists():
            user_dir.mkdir(parents=True)
        file_path = user_dir / f"{conversation_id}.json"
        return FileChatMessageHistory(str(file_path))

    return get_chat_history


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


def _per_request_config_modifier(
    config: Dict[str, Any], request: Request
) -> Dict[str, Any]:
    """Update the config"""
    config = config.copy()
    configurable = config.get("configurable", {})
    # Look for a cookie named "user_id"
    user_id = request.cookies.get("user_id", None)

    if user_id is None:
        raise HTTPException(
            status_code=400,
            detail="No user id found. Please set a cookie named 'user_id'.",
        )

    configurable["user_id"] = user_id
    config["configurable"] = configurable
    return config

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        "You are a professional, caring and emphatetic \
            psychologist who is using a following technique: {technique}. \
            Don't mention the name and the idea behind the technique to the user. \
            Keep your responses as short as possible. \
            Don't give advices to user and don't redirect them to real psychologist nor to hotline. \
            Engage in interactive conversation with user, ask them questions one by one, wait for their response and then ask another question. \
            Make sure the conversation sounds like a real conversation with a psychologist. \
            Don't ever mention that you are a chatbot. \
            Don't back off from the conversation. \
            "
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{human_input}")
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# llm = AzureMLChatOnlineEndpoint(
#     endpoint_url=os.getenv("COHERE_LLM_URL"),
#     endpoint_api_type=AzureMLEndpointApiType.serverless,
#     endpoint_api_key=os.getenv("COHERE_LLM_KEY"),

# )

# llm = AzureChatOpenAI(deployment_name="gpt-35-turbo-16k")

llm = AzureChatOpenAI(deployment_name="gpt-4o", max_tokens=512, temperature=0.4)

retriever = SupabaseRetriever(supabase=supabase, cohere=co)

technique = itemgetter("human_input") | retriever | format_docs

assign = RunnablePassthrough.assign(technique=technique)

chain = assign | prompt | llm


class InputChat(TypedDict):
    """Input for the chat endpoint."""

    human_input: str
    """Human input"""


chain_with_history = RunnableWithMessageHistory(
    chain,
    create_session_factory("chat_histories"),
    input_messages_key="human_input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
).with_types(input_type=InputChat)

@app.get("/conversations/{user_id}", response_class=JSONResponse)
async def get_all_conversations(user_id: str):
    """Get all conversations for a specific user."""
    if not _is_valid_identifier(user_id):
        raise HTTPException(
            status_code=400,
            detail="User ID is not in a valid format. User ID must only contain alphanumeric characters, hyphens, and underscores."
        )
    
    user_dir = Path("chat_histories") / user_id
    if not user_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="User not found."
        )
    
    conversations = {}
    for file_path in user_dir.glob("*.json"):
        conversation_id = file_path.stem
        with open(file_path, "r") as f:
            conversations[conversation_id] = f.read()
    
    return conversations

add_routes(
    app,
    chain_with_history,
    per_req_config_modifier=_per_request_config_modifier,
    disabled_endpoints=["playground", "batch"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)