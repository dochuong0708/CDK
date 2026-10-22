from langchain_core.tools import StructuredTool
from app.models.agent_schema import AudioToolInput
from core.tools.audio_features_tool import audio_features_tool
from core.tools.audio_anomaly_tool import AudioAnomalyTool


class AudioTools:
    def __init__(self, memory, model_path=None):
        self.memory = memory
        self.anomaly = AudioAnomalyTool(model_path)

    def for_session(self, session):
        def features(file_id):
            return audio_features_tool(self.memory.file_path(session, file_id))

        def anomaly(file_id):
            return self.anomaly(self.memory.file_path(session, file_id))

        return [StructuredTool.from_function(features, name="audio_features_tool", args_schema=AudioToolInput,
                    description="Extract MFCC statistics, waveform and mel spectrogram for an uploaded audio file."),
                StructuredTool.from_function(anomaly, name="audio_anomaly_tool", args_schema=AudioToolInput,
                    description="Run the unvalidated ASVspoof CNN and Grad-CAM. Scores do not establish authenticity.")]

    def call(self, session, name, arguments):
        tool = next((t for t in self.for_session(session) if t.name == name), None)
        if tool is None:
            raise ValueError("Tool không nằm trong registry Audio.")
        return tool.invoke(AudioToolInput.model_validate(arguments).model_dump())

    def schemas(self):
        return [dict(name=t.name, description=t.description, parameters=t.args_schema.model_json_schema())
                for t in self.for_session("schema-only")]
