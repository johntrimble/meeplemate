from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from typing import List
import structlog

log = structlog.get_logger(__name__)

class LoggingTracer(BaseTracer):
    name: str = "structlog_callback_handler"

    def _persist_run(self, run: Run) -> None:
        pass

    def get_parents(self, run: Run) -> List[Run]:
        parents = []
        current_run = run
        while current_run.parent_run_id:
            parent = self.run_map.get(str(current_run.parent_run_id))
            if parent:
                parents.append(parent)
                current_run = parent
            else:
                break
        return parents
    
    def get_parent_ids(self, run: Run) -> List[str]:
        return [str(parent.id) for parent in self.get_parents(run)[::-1]]
    
    def _on_run_start(self, run: Run) -> None:
        log.info(
            "run_start",
            run_name=run.name,
            inputs=run.inputs,
            tags=run.tags,
            run_type=run.run_type,
            run_id=str(run.id),
            parent_ids=self.get_parent_ids(run)
        )
    
    def _on_run_end(self, run: Run) -> None:
        log.info(
            "run_end",
            run_name=run.name,
            inputs=run.inputs,
            outputs=run.outputs,
            elapsed=str(run.end_time - run.start_time),
            tags=run.tags,
            run_type=run.run_type,
            run_id=str(run.id),
            parent_ids=self.get_parent_ids(run)
        )
    
    def _on_run_error(self, run: Run) -> None:
        log.error(
            "run_error",
            run_name=run.name,
            error=run.error,
            elapsed=str(run.end_time - run.start_time),
            tags=run.tags,
            run_type=run.run_type,
            run_id=str(run.id),
            parent_ids=self.get_parent_ids(run)
        )
    
    def _on_chain_start(self, run: Run) -> None:
        self._on_run_start(run)
    
    def _on_chain_end(self, run: Run) -> None:
        self._on_run_end(run)
    
    def _on_chain_error(self, run: Run) -> None:
        self._on_run_error(run)

    def _on_llm_start(self, run: Run) -> None:
        self._on_run_start(run)
    
    def _on_llm_end(self, run: Run) -> None:
        self._on_run_end(run)

    def _on_llm_error(self, run: Run) -> None:
        self._on_run_error(run)

    def _on_tool_start(self, run: Run) -> None:
        self._on_run_start(run)

    def _on_tool_end(self, run: Run) -> None:
        self._on_run_end(run)

    def _on_tool_error(self, run: Run) -> None:
        self._on_run_error(run)

