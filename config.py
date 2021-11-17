from dataclasses import asdict, dataclass, field
from typing import Optional

@dataclass
class DataArguments:
    """
    Arguments to data path
    """
    data_path: str = field(
        default="./data",
        metadata={"help": "RE data path"}
    )



@dataclass
class IaaArguments:
    """
    Arguments to evaluate IAA 
    """
    iaa_target: str = field(
        default="pilot",
        metadata={
            "help": "evaluate 'pilot' or 'main' "
        },
    )

    iaa_map_path: Optional[str] = field(
        default="iaa_relation_map.json",
        metadata={
            "help": "relation_map path for IAA"
        },
    )

    iaa_make_output: bool = field(
        default=False,
        metadata={
            "help": "make iaa output file"
        },
    )

    iaa_path: Optional[str] = field(
        default="./iaa/",
        metadata={
            "help": "evaluate 'pilot' or 'main'"
        },
    )

