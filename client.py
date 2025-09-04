import requests
from typing import Dict, Any

class GameClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def new_game(self, scenario: int, player_id: str) -> Dict[str, Any]:
        url = f"{self.base}/new-game"
        params = {"scenario": scenario, "playerId": player_id}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def decide_and_next(self, game_id: str, person_index: int, accept: bool | None) -> Dict[str, Any]:
        url = f"{self.base}/decide-and-next"
        params = {"gameId": game_id, "personIndex": person_index}
        if person_index > 0 or accept is not None:
            params["accept"] = "true" if accept else "false"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
