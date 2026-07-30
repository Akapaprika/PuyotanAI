import sys
from pathlib import Path

_DIST_PATH = Path(__file__).parent.parent / "native" / "dist"
if str(_DIST_PATH) not in sys.path:
    sys.path.insert(0, str(_DIST_PATH))

import puyotan_native as p

def main():
    print("Testing ABS module import...")
    cfg_path = str(Path(__file__).parent.parent / "native" / "resources" / "beam_config.json")
    abs_cfg = p.load_abs_config(cfg_path)
    print(f"Loaded ABS config: depth={abs_cfg.depth}, build_budget={abs_cfg.my_budgets.build}")
    
    match = p.PuyotanMatch(100)
    match.start()
    
    res = p.abs_search(match, 0, abs_cfg)
    print(f"ABS Search Success! Best Action: {res.best_action}, Best Eval: {res.best_eval}")

if __name__ == "__main__":
    main()
