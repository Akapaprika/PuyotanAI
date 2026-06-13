import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(r"c:\Users\FMV\Desktop\アプリ\プログラミング\プロジェクト\PuyotanAI")
sys.path.insert(0, str(PROJECT_ROOT / "native" / "dist"))
sys.path.insert(0, str(PROJECT_ROOT))

import puyotan_native as p
from gui.view_model import PuyotanViewModel
from gui.model import GameModel
from gui.agents import BeamSearchAgent, EmptyPlayerAgent
import gui.config as config

config.VIRTUAL_FRAME_INTERVAL_MS = 0

model = GameModel(seed=999)
vm = PuyotanViewModel(model)

agent = BeamSearchAgent(beam_width=500, look_ahead=3)
vm.set_agent(0, agent)
vm.set_agent(1, EmptyPlayerAgent())

print("Stepping match with BeamSearchAgent and printing history...")
has_chained = False
for frame in range(1000):
    if not model.is_playing():
        print("Game over!")
        break
        
    p1_state = model.match.getPlayer(0)
    act_type = p1_state.current_action.action.type
    
    # If the score changed, print it
    old_score = vm.prev_scores[0]
    vm.update()
    
    new_score = model.match.getPlayer(0).score
    if new_score != old_score:
        print(f"Frame {frame:3d} | Action: {act_type.name:10s} | Score: {old_score:5d} -> {new_score:5d} (+{new_score - old_score:5d}) | CurrentChainScore: {vm.current_chain_scores[0]} | LastChainScore: {vm.last_chain_scores[0]} | ChainCount: {p1_state.chain_count}")
        
    if vm.last_chains[0] > 0 and not has_chained:
        print(f"*** CHAIN FINISHED! Count: {vm.last_chains[0]}, Score: {vm.last_chain_scores[0]}")
        has_chained = True
        # Let's run a bit more to see if it resets or does anything else
        
    if frame > 200 and has_chained:
        break
