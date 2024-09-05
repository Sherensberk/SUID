from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LinkType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OUTPUT: _ClassVar[LinkType]
    INPUT: _ClassVar[LinkType]

class SuportedFrameWorks(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    darknet: _ClassVar[SuportedFrameWorks]
    opencv: _ClassVar[SuportedFrameWorks]
    rknn: _ClassVar[SuportedFrameWorks]
OUTPUT: LinkType
INPUT: LinkType
darknet: SuportedFrameWorks
opencv: SuportedFrameWorks
rknn: SuportedFrameWorks

class NodeId(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class Link(_message.Message):
    __slots__ = ("parent", "type", "id")
    PARENT_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    parent: str
    type: LinkType
    id: int
    def __init__(self, parent: _Optional[str] = ..., type: _Optional[_Union[LinkType, str]] = ..., id: _Optional[int] = ...) -> None: ...

class Connection(_message.Message):
    __slots__ = ("output", "input")
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    output: Link
    input: Link
    def __init__(self, output: _Optional[_Union[Link, _Mapping]] = ..., input: _Optional[_Union[Link, _Mapping]] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ("id", "service", "channel", "callback", "connections")
    ID_FIELD_NUMBER: _ClassVar[int]
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    CALLBACK_FIELD_NUMBER: _ClassVar[int]
    CONNECTIONS_FIELD_NUMBER: _ClassVar[int]
    id: str
    service: str
    channel: str
    callback: str
    connections: _containers.RepeatedCompositeFieldContainer[Connection]
    def __init__(self, id: _Optional[str] = ..., service: _Optional[str] = ..., channel: _Optional[str] = ..., callback: _Optional[str] = ..., connections: _Optional[_Iterable[_Union[Connection, _Mapping]]] = ...) -> None: ...

class Recipe(_message.Message):
    __slots__ = ("id", "nodes")
    ID_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    id: str
    nodes: _containers.RepeatedCompositeFieldContainer[Node]
    def __init__(self, id: _Optional[str] = ..., nodes: _Optional[_Iterable[_Union[Node, _Mapping]]] = ...) -> None: ...

class NoneArgs(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RequestStatus(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: int
    def __init__(self, status: _Optional[int] = ...) -> None: ...

class Image(_message.Message):
    __slots__ = ("width", "height", "data")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    data: bytes
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class NetworkConfig(_message.Message):
    __slots__ = ("framework",)
    FRAMEWORK_FIELD_NUMBER: _ClassVar[int]
    framework: SuportedFrameWorks
    def __init__(self, framework: _Optional[_Union[SuportedFrameWorks, str]] = ...) -> None: ...

class Point2f(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...

class Size2f(_message.Message):
    __slots__ = ("height", "width")
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    height: float
    width: float
    def __init__(self, height: _Optional[float] = ..., width: _Optional[float] = ...) -> None: ...

class Rect(_message.Message):
    __slots__ = ("height", "width", "x", "y")
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    height: int
    width: int
    x: int
    y: int
    def __init__(self, height: _Optional[int] = ..., width: _Optional[int] = ..., x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...

class Probability(_message.Message):
    __slots__ = ("name", "probability")
    CLASS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    name: str
    probability: float
    def __init__(self, name: _Optional[str] = ..., probability: _Optional[float] = ..., **kwargs) -> None: ...

class Network(_message.Message):
    __slots__ = ("cfg", "names", "weights")
    CFG_FIELD_NUMBER: _ClassVar[int]
    NAMES_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    cfg: str
    names: str
    weights: str
    def __init__(self, cfg: _Optional[str] = ..., names: _Optional[str] = ..., weights: _Optional[str] = ...) -> None: ...

class Settings(_message.Message):
    __slots__ = ("driver", "enable_tiles", "include_percentage", "nms", "output_redirection", "snapping", "threshold")
    DRIVER_FIELD_NUMBER: _ClassVar[int]
    ENABLE_TILES_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_PERCENTAGE_FIELD_NUMBER: _ClassVar[int]
    NMS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_REDIRECTION_FIELD_NUMBER: _ClassVar[int]
    SNAPPING_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    driver: int
    enable_tiles: bool
    include_percentage: bool
    nms: float
    output_redirection: bool
    snapping: bool
    threshold: float
    def __init__(self, driver: _Optional[int] = ..., enable_tiles: bool = ..., include_percentage: bool = ..., nms: _Optional[float] = ..., output_redirection: bool = ..., snapping: bool = ..., threshold: _Optional[float] = ...) -> None: ...

class Timestamp(_message.Message):
    __slots__ = ("epoch", "text")
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    epoch: int
    text: str
    def __init__(self, epoch: _Optional[int] = ..., text: _Optional[str] = ...) -> None: ...

class Prediction(_message.Message):
    __slots__ = ("all_probabilities", "best_class", "best_probability", "name", "original_point", "original_size", "object_id", "prediction_index", "rect")
    ALL_PROBABILITIES_FIELD_NUMBER: _ClassVar[int]
    BEST_CLASS_FIELD_NUMBER: _ClassVar[int]
    BEST_PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_POINT_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_SIZE_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_FIELD_NUMBER: _ClassVar[int]
    PREDICTION_INDEX_FIELD_NUMBER: _ClassVar[int]
    RECT_FIELD_NUMBER: _ClassVar[int]
    all_probabilities: _containers.RepeatedCompositeFieldContainer[Probability]
    best_class: int
    best_probability: float
    name: str
    original_point: Point2f
    original_size: Size2f
    object_id: int
    prediction_index: int
    rect: Rect
    def __init__(self, all_probabilities: _Optional[_Iterable[_Union[Probability, _Mapping]]] = ..., best_class: _Optional[int] = ..., best_probability: _Optional[float] = ..., name: _Optional[str] = ..., original_point: _Optional[_Union[Point2f, _Mapping]] = ..., original_size: _Optional[_Union[Size2f, _Mapping]] = ..., object_id: _Optional[int] = ..., prediction_index: _Optional[int] = ..., rect: _Optional[_Union[Rect, _Mapping]] = ...) -> None: ...

class Tiles(_message.Message):
    __slots__ = ("height", "horizontal", "vertical", "width")
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    HORIZONTAL_FIELD_NUMBER: _ClassVar[int]
    VERTICAL_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    height: int
    horizontal: int
    vertical: int
    width: int
    def __init__(self, height: _Optional[int] = ..., horizontal: _Optional[int] = ..., vertical: _Optional[int] = ..., width: _Optional[int] = ...) -> None: ...

class File(_message.Message):
    __slots__ = ("count", "duration", "filename", "original_height", "original_width", "prediction", "resized_height", "resized_width", "tiles")
    COUNT_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_WIDTH_FIELD_NUMBER: _ClassVar[int]
    PREDICTION_FIELD_NUMBER: _ClassVar[int]
    RESIZED_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    RESIZED_WIDTH_FIELD_NUMBER: _ClassVar[int]
    TILES_FIELD_NUMBER: _ClassVar[int]
    count: int
    duration: str
    filename: str
    original_height: int
    original_width: int
    prediction: _containers.RepeatedCompositeFieldContainer[Prediction]
    resized_height: int
    resized_width: int
    tiles: Tiles
    def __init__(self, count: _Optional[int] = ..., duration: _Optional[str] = ..., filename: _Optional[str] = ..., original_height: _Optional[int] = ..., original_width: _Optional[int] = ..., prediction: _Optional[_Iterable[_Union[Prediction, _Mapping]]] = ..., resized_height: _Optional[int] = ..., resized_width: _Optional[int] = ..., tiles: _Optional[_Union[Tiles, _Mapping]] = ...) -> None: ...

class InferenceResult(_message.Message):
    __slots__ = ("file", "network", "settings", "timestamp")
    FILE_FIELD_NUMBER: _ClassVar[int]
    NETWORK_FIELD_NUMBER: _ClassVar[int]
    SETTINGS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    file: _containers.RepeatedCompositeFieldContainer[File]
    network: Network
    settings: Settings
    timestamp: Timestamp
    def __init__(self, file: _Optional[_Iterable[_Union[File, _Mapping]]] = ..., network: _Optional[_Union[Network, _Mapping]] = ..., settings: _Optional[_Union[Settings, _Mapping]] = ..., timestamp: _Optional[_Union[Timestamp, _Mapping]] = ...) -> None: ...

class WorldMappingRequest(_message.Message):
    __slots__ = ("image", "inference")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_FIELD_NUMBER: _ClassVar[int]
    image: Image
    inference: InferenceResult
    def __init__(self, image: _Optional[_Union[Image, _Mapping]] = ..., inference: _Optional[_Union[InferenceResult, _Mapping]] = ...) -> None: ...

class WorldMappingResult(_message.Message):
    __slots__ = ("coordinates",)
    COORDINATES_FIELD_NUMBER: _ClassVar[int]
    coordinates: _containers.RepeatedCompositeFieldContainer[Point2f]
    def __init__(self, coordinates: _Optional[_Iterable[_Union[Point2f, _Mapping]]] = ...) -> None: ...
