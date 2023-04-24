import json
from typing import Any, Dict

from langchain import LLMChain


class JSONChain(LLMChain):
    output_key = "json"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = json.loads(super()._call(inputs)["json"].strip())
        return {"json": data}
