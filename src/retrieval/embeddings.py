"""Amazon Bedrock Titan embedding client with batching and retries."""
import json
from typing import Sequence

import boto3
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings

settings = get_settings()


class BedrockEmbeddings:
    """Embeds text via Titan v2 on Bedrock. Batches up to 25 texts per call."""

    BATCH_SIZE = 25
    DIM = 1024

    def __init__(self) -> None:
        self.client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        self.model_id = settings.bedrock_embedding_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def _embed_single(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text, "dimensions": self.DIM, "normalize": True})
        resp = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
        )
        result = json.loads(resp["body"].read())
        return result["embedding"]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many texts. Titan doesn't support native batching so we loop."""
        return [self._embed_single(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_single(text)
