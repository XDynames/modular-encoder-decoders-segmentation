from pathlib import Path
from typing import Dict, Union

from src.datasets.decode import state_bytes_to_dict


def load_game_state(filepath: Union[Path, str]) -> Dict:
    """
    Loads recorded game state as a dictionary of observations see
        src.game_capture.state.shared_memory.ac for a list of keys.

    :param filepath: Path to game state binary file to be loaded.
    :type filepath: Union[Path,str]
    :return: Game state loaded as a dictionary.
    :rtype: Dict
    """
    with open(filepath, "rb") as file:
        data = file.read()
    return state_bytes_to_dict(data)
