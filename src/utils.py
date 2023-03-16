from pprint import pformat
from typing import Any, Optional, Union


def missing_keys(dict1: Union[dict, Any], dict2: Union[dict, Any]) -> Optional[str]:
    if hasattr(dict1, "__dict__"):
        dict1 = dict1.__dict__
    if hasattr(dict2, "__dict__"):
        dict2 = dict2.__dict__

    keys1 = sorted(dict1.keys())
    keys2 = sorted(dict2.keys())

    keys = list(set(keys1).symmetric_difference(keys2))
    if len(keys) == 0:
        return None
    diffs = []
    diffs.append(f"{keys[0]}: left={dict1[keys[0]]}, right={dict2[keys[0]]}")
    diffs = "\n".join(diffs)
    return (
        f"Objects differ on keys:\n{diffs}\n"
        f"left: {pformat(dict1, indent=2)}\n"
        f"right: {pformat(dict2, indent=2)}\n"
    )
