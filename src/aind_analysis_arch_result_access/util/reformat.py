"""
Some utils for reformatting the data
"""


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
