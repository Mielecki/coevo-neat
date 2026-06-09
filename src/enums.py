from __future__ import annotations

from enum import Enum


class AgentSpecies(str, Enum):

    PREY = "prey"
    PREDATOR = "predator"
