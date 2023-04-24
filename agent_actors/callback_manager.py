from pprint import pprint
from typing import Any, Dict, List

from langchain.callbacks import CallbackManager, StdOutCallbackHandler


class ConsolePrettyPrintManager(CallbackManager):
    def __init__(self, handlers: List[StdOutCallbackHandler] = None):
        super().__init__(handlers=handlers)
        self.handlers.append(ConsolePrettyPrinter())


class ConsolePrettyPrinter(StdOutCallbackHandler):
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        super().on_chain_end(outputs, **kwargs)
        pprint(outputs)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        super().on_chain_start(serialized, inputs, **kwargs)
        pprint(inputs)
