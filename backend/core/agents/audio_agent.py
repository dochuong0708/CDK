import base64
from time import perf_counter
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from app.models.agent_schema import AgentResult
from core.tools.registry import AudioTools


class AudioState(TypedDict, total=False):
    session: str
    file_id: str
    trace: str
    features: dict
    anomaly: dict
    steps: list[dict]
    result: dict
    recalled_memory: list[dict]


class AudioAgent:
    def __init__(self, memory, model_path=None):
        self.memory = memory
        self.tools = AudioTools(memory, model_path)
        graph = StateGraph(AudioState)
        graph.add_node("features", self.features)
        graph.add_node("anomaly", self.anomaly)
        graph.add_node("summary", self.summary)
        graph.add_edge(START, "features")
        graph.add_edge("features", "anomaly")
        graph.add_edge("anomaly", "summary")
        graph.add_edge("summary", END)
        self.graph = graph.compile()

    def execute_tool(self, session, name, arguments):
        result = self.tools.call(session, name, arguments)
        # Keep image bytes out of function outputs and LLM contexts.
        png = result.pop("xaiPngBase64", None)
        if png:
            artifact = str(uuid4())
            directory = self.memory.root / "artifacts" / session
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{artifact}.png").write_bytes(base64.b64decode(png, validate=True))
            result["xaiUrl"] = f"/api/agent/sessions/{session}/artifacts/{artifact}"
        return result

    def _node(self, state, key, name):
        start = perf_counter()
        repo = self.memory.repository
        repo.append_event(state["session"], state["trace"], "tool_started", dict(tool=name, arguments={"file_id": state["file_id"]}))
        try:
            output = self.execute_tool(state["session"], name, {"file_id": state["file_id"]})
        except Exception:
            repo.append_event(state["session"], state["trace"], "tool_failed", dict(tool=name, status="failed", code="tool_execution_failed"))
            raise
        step = dict(tool=name, status=output["status"], executionTimeMs=round((perf_counter()-start)*1000, 2))
        repo.append_event(state["session"], state["trace"], "tool_completed", dict(**step, output=output))
        return {key: output, "steps": state.get("steps", []) + [step]}

    def features(self, state):
        return self._node(state, "features", "audio_features_tool")

    def anomaly(self, state):
        return self._node(state, "anomaly", "audio_anomaly_tool")

    def summary(self, state):
        anomaly = state["anomaly"]
        score = anomaly.get("fakeScore")
        has_score = score is not None
        message = (f"CNN dự đoán {anomaly['predictedClass']}, điểm fake {score:.3f}. " if has_score
                   else "Chưa cấu hình checkpoint; không thể kết luận thật/giả. ")
        message += "Cần kiểm tra thêm: mô hình chưa được xác nhận chất lượng; phổ và Grad-CAM không chứng minh giả mạo."
        result = AgentResult(sessionID=state["session"], traced=state["trace"],
            thoughtProcess=[dict(step=i+1, action=s["tool"], status=s["status"]) for i, s in enumerate(state["steps"])],
            toolsExecuted=state["steps"], label="SUSPICIOUS", riskScore=score if has_score else .5,
            finalResponse=message, evidence=[anomaly["warning"]],
            metadata=dict(runtime="langgraph-explicit-audio-v1", modelStatus=anomaly["modelStatus"],
                riskScoreMeaning="uncalibrated CNN score" if has_score else "0.5 sentinel for unknown; not a model prediction",
                requiresReview=True, features=state["features"], anomaly=anomaly,
                recalledMemory=state.get("recalled_memory", []),
                memoryEncoder="hashing-char-2-4-384; lexical similarity, not semantic embeddings"))
        data = result.model_dump(mode="json")
        self.memory.repository.finish_trace(state["session"], state["trace"], data)
        return {"result": data}

    async def events(self, session, request):
        self.memory.file_path(session, request.file_id)
        recalled = self.memory.search(session, request.message, limit=3)
        repo = self.memory.repository
        trace = repo.start_trace(session, request.model_dump(mode="json"),
                                 ["audio_features_tool", "audio_anomaly_tool"], request.message)
        try:
            yield dict(event="started", data=dict(sessionID=session, traced=trace))
            state = dict(session=session, file_id=str(request.file_id), trace=trace, steps=[], recalled_memory=recalled)
            async for update in self.graph.astream(state, stream_mode="updates"):
                for node, values in update.items():
                    if node == "summary":
                        yield dict(event="result", data=values["result"])
                    else:
                        yield dict(event="tool_completed", data=values["steps"][-1])
        except Exception:
            repo.fail_trace(session, trace, "audio_execution_failed")
            raise
        finally:
            repo.fail_trace(session, trace, "stream_closed", status="cancelled")
