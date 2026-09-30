from streaming.detectors.anticipation import OnlineAnticipation, RetrainedAnticipation
from streaming.detectors.base import Detector, PeerRelative
from streaming.detectors.hst import CapyMOA, RiverHST
from streaming.detectors.nagios import NagiosOracle, NagiosPersistence
from streaming.detectors.rettig import PearsonPeer, RelativeEntropy

KINDS: dict[str, type[Detector]] = {
    "capymoa": CapyMOA,
    "river_hst": RiverHST,
    "relative_entropy": RelativeEntropy,
    "pearson_peer": PearsonPeer,
    "nagios_persistence": NagiosPersistence,
    "nagios_oracle": NagiosOracle,
    "anticipation": OnlineAnticipation,
    "anticipation_gbt": RetrainedAnticipation,
}


def build(spec: dict, features: list[str]) -> Detector:
    params = dict(spec)
    peer_relative = params.pop("peer_relative", False)
    detector = KINDS[params.pop("kind")](features, **params)
    return PeerRelative(detector) if peer_relative else detector
