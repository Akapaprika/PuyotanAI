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

from ai import (
    AgentFactory,
    PlayerMode,
    HumanPlayerAgent,
    EmptyPlayerAgent,
    SoloBeamAgent,
    VsBeamAgent,
    BeamSearchAgent,
    VsBeamSearchAgent,
)


def test_agent_factory_modes():
    modes_all = AgentFactory.get_modes(allow_empty=True)
    modes_no_empty = AgentFactory.get_modes(allow_empty=False)

    # 基本3モードが存在することを確認
    assert AgentFactory.MODE_HUMAN in modes_all
    assert AgentFactory.MODE_AI in modes_all
    assert AgentFactory.MODE_EMPTY_SOLO in modes_all

    # allow_empty=False のとき Empty は除外される
    assert AgentFactory.MODE_EMPTY_SOLO not in modes_no_empty
    assert AgentFactory.MODE_HUMAN in modes_no_empty
    assert AgentFactory.MODE_AI in modes_no_empty


def test_agent_factory_create():
    # PlayerMode Enum direct pass
    agent, err = AgentFactory.create_agent(PlayerMode.HUMAN)
    assert err is None and isinstance(agent, HumanPlayerAgent)

    agent, err = AgentFactory.create_agent(PlayerMode.EMPTY)
    assert err is None and isinstance(agent, EmptyPlayerAgent)

    agent, err = AgentFactory.create_agent(PlayerMode.AI, is_solo=False)
    assert err is None and isinstance(agent, VsBeamAgent)

    agent, err = AgentFactory.create_agent(PlayerMode.AI, is_solo=True)
    assert err is None and isinstance(agent, SoloBeamAgent)

    # String aliases
    agent, err = AgentFactory.create_agent("Human")
    assert err is None and isinstance(agent, HumanPlayerAgent)

    agent, err = AgentFactory.create_agent("AI", is_solo=False)
    assert err is None and isinstance(agent, VsBeamAgent)

    agent, err = AgentFactory.create_agent("AI", is_solo=True)
    assert err is None and isinstance(agent, SoloBeamAgent)

    # 旧モード名の後方互換性確認 (backward-compat aliases)
    agent, err = AgentFactory.create_agent("AI: VS Beam (Gaze / Defense)")
    assert err is None and isinstance(agent, VsBeamSearchAgent)
    assert agent._enable_attack_search is True

    agent, err = AgentFactory.create_agent("AI: VS Beam (No Gaze)")
    assert err is None and isinstance(agent, VsBeamSearchAgent)
    assert agent._enable_attack_search is False

    agent, err = AgentFactory.create_agent("AI: BeamSearch (Solo / Normal)")
    assert err is None and isinstance(agent, BeamSearchAgent)

    # 存在しないモード
    agent, err = AgentFactory.create_agent("NonExistentMode")
    assert agent is None and err is not None


def run_all():
    print("Running test_agent_factory...")
    test_agent_factory_modes()
    test_agent_factory_create()
    print("  [PASS] test_agent_factory")


if __name__ == "__main__":
    run_all()
