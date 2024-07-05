import re
from pathlib import Path
from typing import Any, Callable, Dict, Union

from fastapi import FastAPI, HTTPException, Request
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core import __version__
from langchain import hub
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import ConfigurableFieldSpec, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict
from supabase import create_client, Client
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

prompt = ChatPromptTemplate.from_messages(("system", "You are a psychologist who is using a following technique: {technique}. Keep your responses as short as possible"))

def retriever(prompt):
    user_message = "I want to kill my friend!"

    message_embedding = co.embed(
        texts=[user_message],
        input_type="clustering",
        model="embed-multilingual-v3.0"
    )

    response = supabase.rpc("match_documents", {"query_embedding": message_embedding.embeddings[0]}).execute()

    return prompt.invoke({"technique": response.data[0]['content']})

print(retriever(prompt))


chain = RunnableLambda(retriever(prompt)) | prompt | AzureChatOpenAI(deployment_name="gpt-35-turbo-16k", temperature=1.8, max_tokens=1024)



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


add_routes(
    app,
    chain_with_history,
    per_req_config_modifier=_per_request_config_modifier,
    disabled_endpoints=["playground", "batch"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)