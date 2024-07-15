import json
from dataclasses import dataclass


@dataclass
class GlobalConfig:
    """
    Dataclass to store global configuration settings.

    Attributes:
        paths (str): Paths related to the configuration.
        config (Dict): Dictionary containing configuration settings.
        index_mapping (List[Dict]): List of dictionaries containing index mapping information.

    """

    paths: str
    config: dict
    index_mapping: list[dict]

    def __init__(self,
                 paths: str,
                 config: dict,
                 index_mapping: list[dict]) -> None:
        """
        Initializes a GlobalConfig object with provided attributes.

        Args:
            paths (str): Paths related to the configuration.
            config (Dict): Dictionary containing configuration settings.
            index_mapping (List[Dict]): List of dictionaries containing index mapping information.

        Returns:
            None

        """
        self.paths = paths
        self.config = config
        self.index_mapping = index_mapping

    @classmethod
    def from_json(cls) -> 'GlobalConfig':
        """
        Create a GlobalConfig object from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing configuration data.

        Returns:
            GlobalConfig: A GlobalConfig object initialized from the JSON data.

        """
        with open('../engine/config/static/names.json', 'r') as file:
            paths = json.load(file)

        with open('../' + paths['config.sessionFile'], 'r') as file:
            config = json.load(file)

        with open('../' + paths['config.indexMapping'], 'r') as file:
            index_mapping = json.load(file)
            return cls(
                paths=paths,
                config=config,
                index_mapping=index_mapping
            )
