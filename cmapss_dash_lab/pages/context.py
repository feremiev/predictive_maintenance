from __future__ import annotations

from services import (
    ExperimentService,
    load_project_classes,
)


PROJECT_CLASSES, IMPORT_ERROR = load_project_classes()
SERVICE = ExperimentService(PROJECT_CLASSES)
