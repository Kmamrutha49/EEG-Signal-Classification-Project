COMMAND_MAP = {
    "left": "MOVE LEFT",
    "right": "MOVE RIGHT",
    "blink": "CLICK",
}

def map_command(label):
    return COMMAND_MAP.get(label, "NO ACTION")
