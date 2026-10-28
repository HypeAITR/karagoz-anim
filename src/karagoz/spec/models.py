"""Typed models for every spec file kind (spec v1)."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    ValidationError,
    model_validator,
)
from pydantic_core import PydanticCustomError

from karagoz.spec.anchor import ANCHOR_RE


class SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


Version = Literal[0, 1]
Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$")]
LineId = Annotated[str, Field(pattern=r"^L[0-9]{2,3}$")]
SoundId = Annotated[str, Field(pattern=r"^S[0-9]{2,3}$")]
ShotId = Annotated[str, Field(pattern=r"^sh[0-9]{2,3}[a-z]?$")]
PoseId = Annotated[str, Field(pattern=r"^p[0-9]{1,2}$")]
SequenceId = Annotated[str, Field(pattern=r"^s[0-9]{1,2}$")]
View = Literal[
    "front",
    "three_quarter_left",
    "three_quarter_right",
    "profile_left",
    "profile_right",
    "back",
]
Expression = Literal["neutral", "happy", "sad", "angry", "worried", "surprised"]
Framing = Literal["wide", "medium", "medium_close", "close", "extreme_close", "insert"]
MouthShape = Literal["X", "A", "B", "C", "D", "E", "F", "G", "H"]
Support = Literal["sit", "stand", "crouch", "lie"]
Hand = Literal["left", "right"]


def _check_anchor(v: str) -> str:
    if ANCHOR_RE.fullmatch(v) is None:
        raise PydanticCustomError("anchor_syntax", "not a valid anchor: {value}", {"value": v})
    return v


AnchorStr = Annotated[str, AfterValidator(_check_anchor)]


class LineEntry(SpecModel):
    id: LineId
    speaker: Name
    text: Annotated[str, Field(min_length=1)]
    delivery: str | None = None
    at: AnchorStr
    note: str | None = None


class SoundEntry(SpecModel):
    id: SoundId
    description: str
    at: AnchorStr
    note: str | None = None


class Lines(SpecModel):
    version: Version
    episode: Name
    lines: list[LineEntry]
    sounds: list[SoundEntry] | None = None


class ShotCharacter(SpecModel):
    name: Name
    place: Name | None = None
    facing: Name | None = None
    view: View | None = None
    offscreen: bool | None = None


class Shot(SpecModel):
    id: ShotId
    scene: Annotated[int, Field(ge=1)]
    location: Name
    framing: Framing
    characters: list[ShotCharacter] | None = None
    lines: list[LineId] | None = None
    sounds: list[SoundId] | None = None
    in_: AnchorStr = Field(alias="in")
    out: AnchorStr
    transition_in: Literal["cut", "from_black", "fade_from_black"] | None = None
    transition_out: Literal["cut", "to_black", "fade_to_black"] | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    note: str | None = None


class Shots(SpecModel):
    version: Version
    episode: Name
    shots: Annotated[list[Shot], Field(min_length=1)]


class ContinuityState(SpecModel):
    shot: ShotId
    character: Name
    in_: Annotated[str, Field(min_length=10)] = Field(alias="in")
    out: Annotated[str, Field(min_length=10)]
    note: str | None = None


class PropEndpoint(SpecModel):
    holder: Name | None = None
    place: Name | None = None
    state: Name


class PropEntry(SpecModel):
    shot: ShotId
    prop: Name
    in_: PropEndpoint = Field(alias="in")
    out: PropEndpoint
    note: str | None = None


class Continuity(SpecModel):
    version: Version
    scene: Annotated[int, Field(ge=1)]
    states: list[ContinuityState]
    prop_states: dict[Name, list[Name]] | None = None
    props: list[PropEntry] | None = None


class Pose(SpecModel):
    id: PoseId
    view: View
    expression: Expression | None = None
    acting: Annotated[str, Field(min_length=10)]
    reuse: Annotated[str, Field(pattern=r"^sh[0-9]{2,3}[a-z]?/p[0-9]{1,2}$")] | None = None
    note: str | None = None
    faceless: bool | None = None
    hold: bool | None = None
    support: Support | None = None
    on: Name | None = None
    rests_on: list[Name] | None = None
    holds: dict[Hand, Name | None] | None = None

    @model_validator(mode="after")
    def _expression_unless_faceless(self) -> Self:
        if self.expression is None and self.faceless is not True:
            raise PydanticCustomError("expression_required", "expression is required unless faceless is true")
        return self


class PoseTransition(SpecModel):
    from_: PoseId = Field(alias="from")
    to: PoseId
    inbetweens: Annotated[list[Annotated[str, Field(min_length=10)]], Field(min_length=1, max_length=6)]
    holds: list[dict[Hand, Name | None] | None] | None = None
    note: str | None = None


class PoseSequence(SpecModel):
    id: SequenceId
    from_: PoseId = Field(alias="from")
    to: PoseId | None = None
    drawings: Annotated[list[Annotated[str, Field(min_length=10)]], Field(min_length=2, max_length=48)]
    note: str | None = None


class CharacterPoses(SpecModel):
    poses: Annotated[list[Pose], Field(min_length=1)]
    transitions: list[PoseTransition] | None = None
    sequences: list[PoseSequence] | None = None


class Poses(SpecModel):
    version: Version
    shot: ShotId
    characters: dict[Name, CharacterPoses]


class Hold(SpecModel):
    at: AnchorStr
    pose: PoseId
    note: str | None = None


class Transition(SpecModel):
    at: AnchorStr
    to: PoseId
    via: Literal["inbetween", "snap"]
    cadence: Literal["ones", "twos"] | None = None
    anticipate: bool | None = None
    settle: bool | None = None
    note: str | None = None


class EyeEvent(SpecModel):
    at: AnchorStr
    event: Literal["blink", "close", "open"]
    note: str | None = None


class SequenceEvent(SpecModel):
    at: AnchorStr
    sequence: SequenceId
    cadence: Literal["ones", "twos", "threes"] | None = None
    note: str | None = None


class LoopEvent(SpecModel):
    at: AnchorStr
    loop: Annotated[list[Annotated[str, Field(pattern=r"^(p|s)[0-9]{1,2}$")]], Field(min_length=2)]
    step: Annotated[str, Field(pattern=r"^beat\.[a-z_]+$")] | None = None
    cadence: Literal["ones", "twos", "threes"] | None = None
    until: AnchorStr
    note: str | None = None


def _track_item_tag(value: object) -> str | None:
    if isinstance(value, dict):
        if "pose" in value:
            return "Hold"
        if "to" in value:
            return "Transition"
        if "event" in value:
            return "EyeEvent"
        if "sequence" in value:
            return "SequenceEvent"
        if "loop" in value:
            return "LoopEvent"
    return None


TrackItem = Annotated[
    Annotated[Hold, Tag("Hold")]
    | Annotated[Transition, Tag("Transition")]
    | Annotated[EyeEvent, Tag("EyeEvent")]
    | Annotated[SequenceEvent, Tag("SequenceEvent")]
    | Annotated[LoopEvent, Tag("LoopEvent")],
    Discriminator(_track_item_tag),
]

TrackName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*\.(body|eyes)$")]


class CameraEvent(SpecModel):
    at: AnchorStr
    hold: Framing | None = None
    move: Literal["push_in", "pull_out", "pan", "shake"] | None = None
    target: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*(\.[a-z_]+)?$")] | None = None
    until: AnchorStr | None = None
    ease: Literal["slow", "linear", "fast"] | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _exactly_one_of_hold_or_move(self) -> Self:
        if (self.hold is None) == (self.move is None):
            raise PydanticCustomError("camera_hold_move", "camera event needs exactly one of 'hold' or 'move'")
        return self


class FxEvent(SpecModel):
    at: AnchorStr
    until: AnchorStr | None = None
    kind: Literal["stream", "splash", "drip", "dust", "shake"]
    source: str | None = None
    target: str | None = None
    note: str | None = None


class MouthEdit(SpecModel):
    at: AnchorStr
    shape: MouthShape
    note: str | None = None


class MouthTrack(SpecModel):
    source: Literal["auto", "none"]
    edits: list[MouthEdit] | None = None


class Xsheet(SpecModel):
    version: Version
    shot: ShotId
    tracks: dict[TrackName, list[TrackItem]]
    camera: list[CameraEvent] | None = None
    mouth: dict[Name, MouthTrack] | None = None
    notes: str | None = None
    fx: list[FxEvent] | None = None


class MouthCue(SpecModel):
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(ge=1)]
    shape: MouthShape
    src: Literal["auto", "edit"]
    letters: str | None = None


class Mouth(SpecModel):
    version: Version
    shot: ShotId
    character: Name
    fps: Literal[24]
    note: str | None = None
    cues: list[MouthCue]


class StyledValue(SpecModel):
    value: float
    source: Annotated[str, Field(min_length=3)]
    tune: bool | None = None


StyleGroup = dict[str, StyledValue]


class MouthTableEntry(SpecModel):
    letters: list[Annotated[str, Field(max_length=1)]]
    note: str | None = None


class MouthLive(SpecModel):
    # The schema allows any number of extra {value, source, tune?} entries next to "table".
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    table: dict[MouthShape, MouthTableEntry]

    @model_validator(mode="after")
    def _validate_extra_values(self) -> Self:
        extra = self.model_extra
        if extra:
            for key, raw in extra.items():
                if isinstance(raw, StyledValue):
                    continue
                try:
                    extra[key] = StyledValue.model_validate(raw)
                except ValidationError as exc:
                    raise ValueError(f"invalid value for {key!r}: {exc}") from exc
        return self


class StyleLive(SpecModel):
    beats: StyleGroup
    hold: StyleGroup
    transition: StyleGroup
    mouth: MouthLive
    blink: StyleGroup
    breath: StyleGroup | None = None
    camera: StyleGroup | None = None


class Style(SpecModel):
    version: Version
    fps: Literal[24]
    size: tuple[Literal[1920], Literal[1080]]
    live: StyleLive
    lint: StyleGroup | None = None
    guidance: list[str] | None = None


class Align(SpecModel):
    version: Version
    line: LineId
    text: str
    characters: list[str]
    starts: list[float]
    ends: list[float]

    @model_validator(mode="after")
    def _check_lengths_and_order(self) -> Self:
        if not (len(self.characters) == len(self.starts) == len(self.ends)):
            raise PydanticCustomError("align_lengths", "characters, starts and ends must have equal length")
        for start, end in zip(self.starts, self.ends, strict=True):
            if end < start:
                raise PydanticCustomError("align_order", "every end must be >= its start")
        for prev, cur in zip(self.starts, self.starts[1:], strict=False):
            if cur < prev:
                raise PydanticCustomError("align_order", "starts must be non-decreasing")
        return self


class RegionScale(SpecModel):
    px_per_cm: Annotated[float, Field(gt=0)]
    source: Name
    known_cm: Annotated[float, Field(gt=0)]


class RegionObject(SpecModel):
    name: Name
    kind: Annotated[str, Field(min_length=1)]
    box: tuple[int, int, int, int]
    floor_y: int
    confidence: Annotated[float, Field(ge=0, le=1)]
    mask: Annotated[str, Field(min_length=1)]
    front: bool

    @model_validator(mode="after")
    def _check_box(self) -> Self:
        x0, y0, x1, y1 = self.box
        if x1 <= x0 or y1 <= y0:
            raise PydanticCustomError("regions_box", "box needs x1 > x0 and y1 > y0")
        return self


class Place(SpecModel):
    name: Name
    object: Name
    kind: Literal["seat", "floor"]
    seat: tuple[int, int] | None = None
    floor: tuple[int, int]

    @model_validator(mode="after")
    def _check_seat(self) -> Self:
        if self.kind == "seat" and self.seat is None:
            raise PydanticCustomError("regions_seat", "a place of kind 'seat' needs a seat point")
        return self


class Regions(SpecModel):
    version: Version
    location: Name
    image: Annotated[str, Field(min_length=1)]
    size: tuple[int, int]
    scale: RegionScale | None
    objects: list[RegionObject]
    places: list[Place]

    @model_validator(mode="after")
    def _check_names(self) -> Self:
        object_names = [o.name for o in self.objects]
        if len(set(object_names)) != len(object_names):
            raise PydanticCustomError("regions_names", "duplicate object names")
        place_names = [p.name for p in self.places]
        if len(set(place_names)) != len(place_names):
            raise PydanticCustomError("regions_names", "duplicate place names")
        known = set(object_names)
        for p in self.places:
            if p.object not in known:
                raise PydanticCustomError("regions_names", f"place '{p.name}' has unknown object '{p.object}'")
        return self
