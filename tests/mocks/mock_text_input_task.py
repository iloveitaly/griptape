from attr import define
from griptape.artifacts import TextArtifact
from griptape.tasks import BaseTextInputTask


@define
class MockTextInputTask(BaseTextInputTask):
    def run(self) -> TextArtifact:
        return TextArtifact(self.input.to_text())

    def to_dict(self) -> dict:
        return {}
