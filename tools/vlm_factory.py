import logging

from openai import AzureOpenAI
from azure.identity import ClientSecretCredential

from ..config import get_config
from ..exceptions import ToolExecutionError

logger = logging.getLogger(__name__)


class AzureOpenAIVLM:

    def __init__(self):
        cfg = get_config().openai

        credential = ClientSecretCredential(
            tenant_id=cfg.tenant_id,
            client_id=cfg.client_id,
            client_secret=cfg.client_secret,
        )

        # azure-identity caches tokens internally — this is efficient
        def _token_provider():
            return credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            ).token

        self.client = AzureOpenAI(
            azure_endpoint=cfg.endpoint,
            azure_ad_token_provider=_token_provider,
            api_version=cfg.api_version,
        )
        self._deployment = cfg.deployment
        self._temperature = cfg.temperature

    def analyze_image(self, prompt: str, image_base64: str) -> str:
        """Send an image + prompt to the vision model."""
        try:
            response = self.client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            }
                        ],
                    },
                ],
                temperature=self._temperature,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.error("VLM image analysis failed: %s", exc)
            raise ToolExecutionError(str(exc)) from exc

    def analyze_text(
        self,
        system_prompt: str,
        user_content: str,
        *,
        response_format: dict | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a text-only prompt. Used by interpretation agent."""
        try:
            kwargs: dict = {
                "model": self._deployment,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature if temperature is not None else self._temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as exc:
            logger.error("VLM text analysis failed: %s", exc)
            raise ToolExecutionError(str(exc)) from exc
