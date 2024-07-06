from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from pydantic import BaseModel

from supabase import Client
from cohere import Client as CohereClient


class SupabaseRetriever(BaseRetriever):

    supabase: Client

    cohere: CohereClient

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # Get the relevant documents from the database

        message_embeddings = self.cohere.embed(
            texts = [query],
            input_type = "clustering",
            model="embed-multilingual-v3.0"
        )

        response = self.supabase.rpc("match_documents", {"query_embedding": message_embeddings.embeddings[0]}).execute()

        return [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in response.data]