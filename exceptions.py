class DocumentProcessingError(Exception):
    pass


class OCRFailure(DocumentProcessingError):
    pass


class ToolExecutionError(DocumentProcessingError):
    pass


class AgentExecutionError(DocumentProcessingError):
    pass
