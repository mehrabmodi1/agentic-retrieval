from __future__ import annotations

import itertools
from typing import Any

from agent_retrieval.schema.template import ExperimentTemplate, GridSpec, Parametrisation


def expand_gridspec(grid: GridSpec, experiment_type: str) -> list[Parametrisation]:
    dimensions: list[tuple[str, list[Any]]] = [
        ("content_profile", grid.content_profile),
        ("corpus_token_count", grid.corpus_token_count),
        ("discriminability", grid.discriminability),
        ("reference_clarity", grid.reference_clarity),
    ]
    if grid.n_items is not None:
        dimensions.append(("n_items", grid.n_items))

    keys = [k for k, _ in dimensions]
    values = [v for _, v in dimensions]

    parametrisations: list[Parametrisation] = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        params["experiment_type"] = experiment_type
        parametrisations.append(Parametrisation(**params))

    return parametrisations


def expand_grid(template: ExperimentTemplate) -> list[Parametrisation]:
    return expand_gridspec(template.grid, template.experiment_type)


def filter_parametrisations(
    parametrisations: list[Parametrisation],
    filters: dict[str, list[Any]],
) -> list[Parametrisation]:
    if not filters:
        return parametrisations

    result: list[Parametrisation] = []
    for p in parametrisations:
        match = True
        for key, allowed_values in filters.items():
            if getattr(p, key) not in allowed_values:
                match = False
                break
        if match:
            result.append(p)
    return result
