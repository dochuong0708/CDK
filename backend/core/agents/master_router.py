"""Content-type router with conditional LangGraph edges; no LLM calls or claims."""
from time import perf_counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from app.models.agent_schema import AgentResult
from app.detectors.image.image_detector import ImageDetector
from app.detectors.language.language_detector import LanguageDetector


class MasterState(TypedDict, total=False):
    session: str
    trace: str
    tasks: list[dict]
    index: int
    outputs: list[dict]
    steps: list[dict]
    result: dict


class MasterRouter:
    def __init__(self, audio_agent):
        self.audio = audio_agent
        self.memory = audio_agent.memory
        graph = StateGraph(MasterState)
        graph.add_node("execute", self.execute)
        graph.add_node("summary", self.summary)
        graph.add_edge(START, "execute")
        graph.add_conditional_edges("execute", lambda s: "execute" if s["index"] < len(s["tasks"]) else "summary",
                                    {"execute": "execute", "summary": "summary"})
        graph.add_edge("summary", END)
        self.graph = graph.compile()

    def plan(self, session, request):
        tasks = []
        if request.text.strip():
            tasks.append(dict(tool="text_heuristic_tool", arguments=dict(text=request.text), modality="text"))
        for file_id in request.file_ids:
            kind = self.memory.file_kind(session, file_id)
            if kind == "audio":
                names = ("audio_features_tool", "audio_anomaly_tool")
            elif kind == "image":
                names = ("image_heuristic_tool",)
            else:
                raise ValueError("Không có tool cho loại file này.")
            tasks.extend(dict(tool=name, arguments=dict(file_id=str(file_id)), modality=kind) for name in names)
        return tasks

    def execute(self, state):
        task = state["tasks"][state["index"]]
        name, arguments = task["tool"], task["arguments"]
        repo = self.memory.repository
        repo.append_event(state["session"], state["trace"], "tool_started", task)
        start = perf_counter()
        try:
            if name.startswith("audio_"):
                output = self.audio.execute_tool(state["session"], name, arguments)
            else:
                if name == "text_heuristic_tool":
                    analysis = LanguageDetector().analyze(arguments["text"])
                elif name == "image_heuristic_tool":
                    analysis = ImageDetector().analyze(self.memory.file_path(state["session"], arguments["file_id"]))
                else:
                    raise ValueError("Tool không được đăng ký.")
                output = dict(status="ok", modelStatus="legacy_heuristic", rawAnalysis=analysis.model_dump(mode="json"),
                              riskScore=analysis.riskScore/100, detector=analysis.detector,
                              note="Heuristic cũ chỉ phục vụ tích hợp; chưa phải Vision/NLP detector đã xác nhận chất lượng.")
        except Exception:
            repo.append_event(state["session"], state["trace"], "tool_failed", dict(tool=name, status="failed", code="tool_execution_failed"))
            raise
        step = dict(tool=name, status=output["status"], executionTimeMs=round((perf_counter()-start)*1000, 2))
        item = dict(**step, modality=task["modality"], arguments=arguments, output=output)
        repo.append_event(state["session"], state["trace"], "tool_completed", item)
        return dict(index=state["index"]+1, outputs=state["outputs"]+[item], steps=state["steps"]+[step])

    def summary(self, state):
        scores = [x["output"].get("fakeScore", x["output"].get("riskScore")) for x in state["outputs"]]
        scores = [s for s in scores if s is not None]
        data = AgentResult(sessionID=state["session"], traced=state["trace"],
            thoughtProcess=[dict(step=i+1, action=t["tool"], status=t["status"]) for i,t in enumerate(state["steps"])],
            toolsExecuted=state["steps"], label="SUSPICIOUS", riskScore=max(scores) if scores else .5,
            finalResponse=f"Đã gọi {len(state['steps'])} tool theo loại đầu vào. Các mô hình hiện tại chưa được xác nhận chất lượng; cần kiểm tra thêm trước khi kết luận.",
            evidence=["Router chọn tool theo nội dung text và loại file đã được backend kiểm tra.",
                      "Điểm tổng hợp là max điểm thô để hiển thị demo, không phải xác suất giả mạo."],
            metadata=dict(runtime="langgraph-master-router-v1", routing="validated-input-type", llmUsed=False,
                requiresReview=True, modelStatus="unvalidated", outputs=state["outputs"],
                riskScoreMeaning="maximum uncalibrated score; 0.5 unknown sentinel when all unavailable")).model_dump(mode="json")
        self.memory.repository.finish_trace(state["session"], state["trace"], data)
        return dict(result=data)

    async def events(self, session, request):
        tasks = self.plan(session, request)
        repo = self.memory.repository
        trace = repo.start_trace(session, request.model_dump(mode="json"), [t["tool"] for t in tasks],
                                 request.message + ("\n"+request.text if request.text.strip() else ""))
        try:
            yield dict(event="started", data=dict(sessionID=session, traced=trace))
            repo.append_event(session, trace, "routed", dict(tasks=tasks))
            yield dict(event="routed", data=dict(tasks=tasks))
            state = dict(session=session, trace=trace, tasks=tasks, index=0, outputs=[], steps=[])
            async for update in self.graph.astream(state, stream_mode="updates", config={"recursion_limit": 20}):
                if "summary" in update:
                    yield dict(event="result", data=update["summary"]["result"])
                else:
                    yield dict(event="tool_completed", data=update["execute"]["steps"][-1])
        except Exception:
            repo.fail_trace(session, trace, "master_execution_failed")
            raise
        finally:
            repo.fail_trace(session, trace, "stream_closed", status="cancelled")
