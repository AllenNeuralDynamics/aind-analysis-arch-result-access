"""
Some utils for reformatting the data
"""

import re


def trainer_mapper(trainer):
    """
    Normalize trainer name
    """
    user_mapper = {
        "Avalon Amaya": ["Avalon"],
        "Ella Hilton": ["Ella"],
        "Katrina Nguyen": ["Katrina"],
        "Lucas Kinsey": ["Lucas"],
        "Travis Ramirez": ["Travis"],
        "Xinxin Yin": ["Xinxin", "the ghost"],
        "Bowen Tan": ["Bowen"],
        "Henry Loeffler": ["Henry Loeffer"],
        "Margaret Lee": ["margaret lee"],
        "Madeline Tom": ["Madseline Tom"],
    }
    for canonical_name, alias in user_mapper.items():
        for key_word in alias:
            if key_word in trainer:
                return canonical_name
    else:
        return trainer


def data_source_mapper(rig):
    """From rig string, return "{institute}_{rig_type}_{room}_{hardware}" """

    institute = "Janelia" if ("bpod" in rig) and not ("AIND" in rig) else "AIND"
    hardware = "bpod" if ("bpod" in rig) else "bonsai"
    rig_type = "ephys" if ("ephys" in rig.lower()) else "training"

    # This is a mess...
    if institute == "Janelia":
        room = "NA"
    elif "Ephys-Han" in rig:
        room = "321"
    elif hardware == "bpod":
        room = "347"
    elif "447" in rig:
        room = "447"
    elif "446" in rig:
        room = "446"
    elif "323" in rig:
        room = "323"
    elif "322" in rig:
        room = "322"
    elif rig_type == "ephys":
        room = "323"
    else:
        room = "447"
    return (
        institute,
        rig_type,
        room,
        hardware,
        "_".join([institute, rig_type, room, hardware]),
    )


def split_nwb_name(nwb_name):
    """Turn the nwb_name into subject_id, session_date, nwb_suffix in order to be merged to
    the main df.

    Parameters
    ----------
    nwb_name : str. The name of the nwb file. This function can handle the following formats:
        "721403_2024-08-09_08-39-12.nwb"
        "685641_2023-10-04.nwb",
        "behavior_754280_2024-11-14_11-06-24.nwb",
        "behavior_1_2024-08-05_15-48-54",
        "
        ...

    Returns
    -------
    subject_id : str. The subject ID
    session_date : str. The session date
    nwb_suffix : int. The nwb suffix (converted from session time if available, otherwise 0)
    """

    pattern = r"(?:\w+_)?(?P<subject_id>\d+)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<time>\d{2}-\d{2}-\d{2}))?(?:.*)?"  # noqa E501
    matches = re.search(pattern, nwb_name)

    if not matches:
        return None, None, 0

    subject_id = matches.group("subject_id")
    session_date = matches.group("date")
    session_time = matches.group("time")
    if session_time:
        nwb_suffix = int(session_time.replace("-", ""))
    else:
        nwb_suffix = 0

    return subject_id, session_date, nwb_suffix


def curriculum_ver_mapper(ver):
    """Manually group curriculum versions:
        ver < 2.0 --> v1
        2.0 <= ver < 2.3 --> v2
        ver >=2.3 --> v3

    See discussion here: https://hanhou.notion.site/notes-on-motivation
    """
    if not isinstance(ver, str):
        return None
    if "2.3" in ver:
        return "v3"
    elif "1.0" in ver:
        return "v1"
    else:
        return "v2"
