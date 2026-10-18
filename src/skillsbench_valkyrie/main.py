"""ASGI entrypoint for the SkillsBench Valkyrie benchmark service."""

from benchmark_service.vals import ValsBenchmarkServiceApp as BenchmarkServiceApp
from fastapi import HTTPException, Query, Request
from fastapi.responses import JSONResponse

from skillsbench_valkyrie.service import SkillsBenchBenchmarkService

app = BenchmarkServiceApp(SkillsBenchBenchmarkService)


def _remove_route(path: str) -> None:
    app.router.routes = [route for route in app.router.routes if getattr(route, "path", None) != path]


@app.get("/retrieve-task/")
async def retrieve_task_compat(
    request: Request,
    task_id: str = Query(..., description="Task ID to retrieve"),
    skip_validation: bool = Query(False, description="Skip validation of task existence"),
    dataset: str | None = Query(default=None, description="Dataset name to use (defaults to 'default')"),
) -> JSONResponse:
    """Return both current and legacy task metadata for hosted Valkyrie workers."""
    if not await app.service.check_dataset_access(request.state.tenant, dataset):
        raise HTTPException(status_code=403, detail="Dataset not allowed")

    response = await app.service.retrieve_task(task_id, skip_validation, dataset=dataset)
    payload = response.model_dump(mode="json")
    source = payload.get("source", {})
    if isinstance(source, dict) and source.get("type") == "image":
        payload["docker_image"] = source.get("image")
    return JSONResponse(payload)


_remove_route("/retrieve-task/")
app.add_api_route("/retrieve-task/", retrieve_task_compat, methods=["GET"])
