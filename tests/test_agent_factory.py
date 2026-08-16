"""
tests/test_agent_factory.py
AgentFactory の動作検証テスト。
"""
import sys
from pathlib import Path

_DIST_PATH = Path(__file__).parent.parent / "native" / "dist"
_PROJECT_ROOT = Path(__file__).parent.parent

if str(_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(_DIST_PATH))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gui.agent_factory import AgentFactory
from gui.agents import (
    HumanPlayerAgent,
    EmptyPlayerAgent,
    BeamSearchAgent,
    VsBeamSearchAgent,
    NegamaxAgent,
)


def test_agent_factory_modes():
    modes_all = AgentFactory.get_modes(allow_empty=True)
    modes_no_empty = AgentFactory.get_modes(allow_empty=False)

    assert AgentFactory.MODE_EMPTY_SOLO in modes_all
    assert AgentFactory.MODE_EMPTY_SOLO not in modes_no_empty
    assert AgentFactory.MODE_HUMAN in modes_all
    assert AgentFactory.MODE_NEGAMAX in modes_all


def test_agent_factory_default_config():
    defaults = AgentFactory.get_default_config()
    assert "width" in defaults and defaults["width"] > 0
    assert "depth" in defaults and defaults["depth"] > 0
    assert "dbs" in defaults and defaults["dbs"] >= 0


def test_agent_factory_create():
    # Human
    agent, err = AgentFactory.create_agent(AgentFactory.MODE_HUMAN)
    assert err is None and isinstance(agent, HumanPlayerAgent)

    # Empty
    agent, err = AgentFactory.create_agent(AgentFactory.MODE_EMPTY_SOLO)
    assert err is None and isinstance(agent, EmptyPlayerAgent)

    # BeamSearch
    agent, err = AgentFactory.create_agent(AgentFactory.MODE_BEAM_SEARCH_PLAYER, width=500, depth=3, dbs=4)
    assert err is None and isinstance(agent, BeamSearchAgent)

    # VsBeam (Attack ON)
    agent, err = AgentFactory.create_agent(AgentFactory.MODE_NEW_AI_ATTACK_ON, width=500, depth=3, dbs=4)
    assert err is None and isinstance(agent, VsBeamSearchAgent)
    assert agent._enable_attack_search is True

    # VsBeam (Attack OFF)
    agent, err = AgentFactory.create_agent(AgentFactory.MODE_OLD_AI_ATTACK_OFF, width=500, depth=3, dbs=4)
    assert err is None and isinstance(agent, VsBeamSearchAgent)
    assert agent._enable_attack_search is False

    # Negamax
    agent, err = AgentFactory.create_agent(AgentFactory.MODE_NEGAMAX, width=500, depth=3, dbs=4)
    assert err is None and isinstance(agent, NegamaxAgent)

    # Invalid mode
    agent, err = AgentFactory.create_agent("NonExistentMode")
    assert agent is None and err is not None


def run_all():
    print("Running test_agent_factory...")
    test_agent_factory_modes()
    test_agent_factory_default_config()
    test_agent_factory_create()
    print("  [PASS] test_agent_factory")


if __name__ == "__main__":
    run_all()
