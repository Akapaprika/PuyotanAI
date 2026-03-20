#include <iostream>
#include <puyotan/game/puyotan_match.hpp>

int main() {
    std::cout << "Starting 1 game in C++..." << std::endl;
    puyotan::PuyotanMatch match(1);
    match.start();

    int p1_move = 0;
    int p2_move = 0;
    int loop_count = 0;
    
    // 6 at col 5, 6 at 4, 6 at 3, etc.
    const int move_plan[] = {
        5,5,5,5,5,5,
        4,4,4,4,4,4,
        3,3,3,3,3,3
    };
    const int num_moves = sizeof(move_plan) / sizeof(move_plan[0]);

    while (match.getStatus() == puyotan::MatchStatus::PLAYING) {
        if (++loop_count > 10000) {
            std::cout << "HANG DETECTED! frame=" << match.getFrame() << std::endl;
            auto const& p1 = match.getPlayer(0);
            auto const& p2 = match.getPlayer(1);
            std::cout << "P1 action type=" << (int)p1.action_histories[match.getFrame() & 255].action.type << std::endl;
            std::cout << "P2 action type=" << (int)p2.action_histories[match.getFrame() & 255].action.type << std::endl;
            break;
        }

        // We can't use match.players_ here because it's private, so we use getPlayer
        // wait, we can't mutate getPlayer because it's const.
        // We'll just call setAction if type is NONE
        auto p1_type = match.getPlayer(0).action_histories[match.getFrame() & 255].action.type;
        auto p2_type = match.getPlayer(1).action_histories[match.getFrame() & 255].action.type;

        if (p1_type == puyotan::ActionType::NONE) {
            int col = (p1_move < num_moves) ? move_plan[p1_move] : 2;
            if (match.setAction(0, puyotan::Action{puyotan::ActionType::PUT, static_cast<int8_t>(col), puyotan::Rotation::Up})) {
                p1_move++;
            }
        }
        if (p2_type == puyotan::ActionType::NONE) {
            int col = (p2_move < num_moves) ? move_plan[p2_move] : 2;
            if (match.setAction(1, puyotan::Action{puyotan::ActionType::PUT, static_cast<int8_t>(col), puyotan::Rotation::Up})) {
                p2_move++;
            }
        }

        if (match.canStepNextFrame()) {
            match.stepNextFrame();
        } else {
            // Wait, why would it not step?!
            std::cout << "canStepNextFrame is FALSE! p1_type=" << (int)p1_type << " p2_type=" << (int)p2_type << " frame=" << match.getFrame() << std::endl;
            // Print the newly set types
            auto p1_new = match.getPlayer(0).action_histories[match.getFrame() & 255].action.type;
            auto p2_new = match.getPlayer(1).action_histories[match.getFrame() & 255].action.type;
            std::cout << "After setAction, p1_new=" << (int)p1_new << " p2_new=" << (int)p2_new << std::endl;
            break;
        }
    }
    std::cout << "Finished! Final frame=" << match.getFrame() << " status=" << (int)match.getStatus() << std::endl;
    return 0;
}
