from enum import IntEnum


class TauID(IntEnum):
    BACKGROUND = 0
    HADRONIC_DM0 = 1
    HADRONIC_OTHER = 2
    ELECTRON = 3
    MUON = 4


TAU_ID_CLASS_NAMES = (
    "background", "hadronic_dm0", "hadronic_other", "tau_electron", "tau_muon"
)