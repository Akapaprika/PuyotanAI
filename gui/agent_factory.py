"""
gui/agent_factory.py

Backward-compatible re-export module. AgentFactory is now located in `ai.AgentFactory`.
"""
from ai import AgentFactory, AgentDefaults

__all__ = ["AgentFactory", "AgentDefaults"]
