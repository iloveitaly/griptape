import json
from typing import Optional
from attr import define, field
from griptape.artifacts import BaseArtifact
from griptape.tokenizers import BaseTokenizer


@define(frozen=True)
class TextOutput(BaseArtifact):
    value: Optional[str] = field()

    def token_count(self, tokenizer: BaseTokenizer) -> Optional[int]:
        if isinstance(self.value, str):
            return tokenizer.token_count(self.value)
        else:
            return None

    def __str__(self):
        from griptape.schemas import TextOutputSchema

        return json.dumps(TextOutputSchema().dump(self))
