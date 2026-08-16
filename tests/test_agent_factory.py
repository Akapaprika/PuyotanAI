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
    HumanPlayerAgent,
    EmptyPlayerAgent,
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
    # Human
    agent, err = AgentFactory.create_agent(AgentFactory.MODE_HUMAN)
    assert err is None and isinstance(agent, HumanPlayerAgent)

    # Empty
    agent, err = AgentFactory.create_agent(AgentFactory.MODE_EMPTY_SOLO)
    assert err is None and isinstance(agent, EmptyPlayerAgent)

    # AI (vs beam search with attack search ON)
    agent, err = AgentFactory.create_agent(AgentFactory.MODE_AI)
    assert err is None and isinstance(agent, VsBeamSearchAgent)
    assert agent._enable_attack_search is True

    # 旧モード名の後方互換性確認 (backward-compat aliases)
    agent, err = AgentFactory.create_agent("AI: VS Beam (Gaze / Defense)")
    assert err is None and isinstance(agent, VsBeamSearchAgent)
    assert agent._enable_attack_search is True

    agent, err = AgentFactory.create_agent("AI: VS Beam (No Gaze)")
    assert err is None and isinstance(agent, VsBeamSearchAgent)
    assert agent._enable_attack_search is False

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
