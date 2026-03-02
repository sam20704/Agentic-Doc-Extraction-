import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AzureDIConfig:
    endpoint: str = ""
    api_key: str = ""
    model_id: str = "prebuilt-layout"

    def __post_init__(self):
        self.endpoint = self.endpoint or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
        self.api_key = self.api_key or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
        self.model_id = self.model_id or os.getenv("AZURE_DOC_INTELLIGENCE_MODEL", "prebuilt-layout")


@dataclass
class AzureOpenAIConfig:
    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    endpoint: str = ""
    deployment: str = ""
    api_version: str = "2024-02-15-preview"
    temperature: float = 0.0

    def __post_init__(self):
        self.tenant_id = self.tenant_id or os.getenv("AZURE_TENANT_ID", "")
        self.client_id = self.client_id or os.getenv("AZURE_CLIENT_ID", "")
        self.client_secret = self.client_secret or os.getenv("AZURE_CLIENT_SECRET", "")
        self.endpoint = self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.deployment = self.deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        self.api_version = self.api_version or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
        )
        self.temperature = self.temperature or float(os.getenv("LLM_TEMPERATURE", "0"))


@dataclass
class AppConfig:
    di: AzureDIConfig = field(default_factory=AzureDIConfig)
    openai: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)


_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
