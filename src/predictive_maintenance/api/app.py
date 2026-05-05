from fastapi import FastAPI

from predictive_maintenance.api.routes import router


app = FastAPI(title="Predictive Maintenance API")
app.include_router(router)
