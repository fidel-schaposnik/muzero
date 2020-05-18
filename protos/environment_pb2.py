# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/environment.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow.core.framework import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/environment.proto',
  package='tensorflow.muzero',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x18protos/environment.proto\x12\x11tensorflow.muzero\x1a&tensorflow/core/framework/tensor.proto\"]\n\x05State\x12,\n\x0bobservation\x18\x01 \x01(\x0b\x32\x17.tensorflow.TensorProto\x12\x0f\n\x07to_play\x18\x02 \x01(\x05\x12\x15\n\rlegal_actions\x18\x03 \x03(\x05\"\xba\x01\n\x15InitializationRequest\x12\x63\n\x16\x65nvironment_parameters\x18\x01 \x03(\x0b\x32\x43.tensorflow.muzero.InitializationRequest.EnvironmentParametersEntry\x1a<\n\x1a\x45nvironmentParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\"\x89\x01\n\x16InitializationResponse\x12\x16\n\x0e\x65nvironment_id\x18\x01 \x01(\x05\x12\x19\n\x11\x61\x63tion_space_size\x18\x02 \x01(\x05\x12\x13\n\x0bnum_players\x18\x03 \x01(\x05\x12\'\n\x05state\x18\x04 \x01(\x0b\x32\x18.tensorflow.muzero.State\"6\n\rActionRequest\x12\x16\n\x0e\x65nvironment_id\x18\x01 \x01(\x05\x12\r\n\x05index\x18\x02 \x01(\x05\"h\n\x0e\x41\x63tionResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0e\n\x06reward\x18\x02 \x01(\x02\x12\'\n\x05state\x18\x03 \x01(\x0b\x32\x18.tensorflow.muzero.State\x12\x0c\n\x04\x64one\x18\x05 \x01(\x08\x32\xcb\x01\n\x11RemoteEnvironment\x12g\n\x0eInitialization\x12(.tensorflow.muzero.InitializationRequest\x1a).tensorflow.muzero.InitializationResponse\"\x00\x12M\n\x04Step\x12 .tensorflow.muzero.ActionRequest\x1a!.tensorflow.muzero.ActionResponse\"\x00\x62\x06proto3')
  ,
  dependencies=[tensorflow_dot_core_dot_framework_dot_tensor__pb2.DESCRIPTOR,])




_STATE = _descriptor.Descriptor(
  name='State',
  full_name='tensorflow.muzero.State',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='observation', full_name='tensorflow.muzero.State.observation', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='to_play', full_name='tensorflow.muzero.State.to_play', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='legal_actions', full_name='tensorflow.muzero.State.legal_actions', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=87,
  serialized_end=180,
)


_INITIALIZATIONREQUEST_ENVIRONMENTPARAMETERSENTRY = _descriptor.Descriptor(
  name='EnvironmentParametersEntry',
  full_name='tensorflow.muzero.InitializationRequest.EnvironmentParametersEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.muzero.InitializationRequest.EnvironmentParametersEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.muzero.InitializationRequest.EnvironmentParametersEntry.value', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=309,
  serialized_end=369,
)

_INITIALIZATIONREQUEST = _descriptor.Descriptor(
  name='InitializationRequest',
  full_name='tensorflow.muzero.InitializationRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='environment_parameters', full_name='tensorflow.muzero.InitializationRequest.environment_parameters', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_INITIALIZATIONREQUEST_ENVIRONMENTPARAMETERSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=183,
  serialized_end=369,
)


_INITIALIZATIONRESPONSE = _descriptor.Descriptor(
  name='InitializationResponse',
  full_name='tensorflow.muzero.InitializationResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='environment_id', full_name='tensorflow.muzero.InitializationResponse.environment_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='action_space_size', full_name='tensorflow.muzero.InitializationResponse.action_space_size', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_players', full_name='tensorflow.muzero.InitializationResponse.num_players', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='state', full_name='tensorflow.muzero.InitializationResponse.state', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=372,
  serialized_end=509,
)


_ACTIONREQUEST = _descriptor.Descriptor(
  name='ActionRequest',
  full_name='tensorflow.muzero.ActionRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='environment_id', full_name='tensorflow.muzero.ActionRequest.environment_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='index', full_name='tensorflow.muzero.ActionRequest.index', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=511,
  serialized_end=565,
)


_ACTIONRESPONSE = _descriptor.Descriptor(
  name='ActionResponse',
  full_name='tensorflow.muzero.ActionResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='tensorflow.muzero.ActionResponse.success', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reward', full_name='tensorflow.muzero.ActionResponse.reward', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='state', full_name='tensorflow.muzero.ActionResponse.state', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='done', full_name='tensorflow.muzero.ActionResponse.done', index=3,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=567,
  serialized_end=671,
)

_STATE.fields_by_name['observation'].message_type = tensorflow_dot_core_dot_framework_dot_tensor__pb2._TENSORPROTO
_INITIALIZATIONREQUEST_ENVIRONMENTPARAMETERSENTRY.containing_type = _INITIALIZATIONREQUEST
_INITIALIZATIONREQUEST.fields_by_name['environment_parameters'].message_type = _INITIALIZATIONREQUEST_ENVIRONMENTPARAMETERSENTRY
_INITIALIZATIONRESPONSE.fields_by_name['state'].message_type = _STATE
_ACTIONRESPONSE.fields_by_name['state'].message_type = _STATE
DESCRIPTOR.message_types_by_name['State'] = _STATE
DESCRIPTOR.message_types_by_name['InitializationRequest'] = _INITIALIZATIONREQUEST
DESCRIPTOR.message_types_by_name['InitializationResponse'] = _INITIALIZATIONRESPONSE
DESCRIPTOR.message_types_by_name['ActionRequest'] = _ACTIONREQUEST
DESCRIPTOR.message_types_by_name['ActionResponse'] = _ACTIONRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

State = _reflection.GeneratedProtocolMessageType('State', (_message.Message,), dict(
  DESCRIPTOR = _STATE,
  __module__ = 'protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muzero.State)
  ))
_sym_db.RegisterMessage(State)

InitializationRequest = _reflection.GeneratedProtocolMessageType('InitializationRequest', (_message.Message,), dict(

  EnvironmentParametersEntry = _reflection.GeneratedProtocolMessageType('EnvironmentParametersEntry', (_message.Message,), dict(
    DESCRIPTOR = _INITIALIZATIONREQUEST_ENVIRONMENTPARAMETERSENTRY,
    __module__ = 'protos.environment_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.muzero.InitializationRequest.EnvironmentParametersEntry)
    ))
  ,
  DESCRIPTOR = _INITIALIZATIONREQUEST,
  __module__ = 'protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muzero.InitializationRequest)
  ))
_sym_db.RegisterMessage(InitializationRequest)
_sym_db.RegisterMessage(InitializationRequest.EnvironmentParametersEntry)

InitializationResponse = _reflection.GeneratedProtocolMessageType('InitializationResponse', (_message.Message,), dict(
  DESCRIPTOR = _INITIALIZATIONRESPONSE,
  __module__ = 'protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muzero.InitializationResponse)
  ))
_sym_db.RegisterMessage(InitializationResponse)

ActionRequest = _reflection.GeneratedProtocolMessageType('ActionRequest', (_message.Message,), dict(
  DESCRIPTOR = _ACTIONREQUEST,
  __module__ = 'protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muzero.ActionRequest)
  ))
_sym_db.RegisterMessage(ActionRequest)

ActionResponse = _reflection.GeneratedProtocolMessageType('ActionResponse', (_message.Message,), dict(
  DESCRIPTOR = _ACTIONRESPONSE,
  __module__ = 'protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muzero.ActionResponse)
  ))
_sym_db.RegisterMessage(ActionResponse)


_INITIALIZATIONREQUEST_ENVIRONMENTPARAMETERSENTRY._options = None

_REMOTEENVIRONMENT = _descriptor.ServiceDescriptor(
  name='RemoteEnvironment',
  full_name='tensorflow.muzero.RemoteEnvironment',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=674,
  serialized_end=877,
  methods=[
  _descriptor.MethodDescriptor(
    name='Initialization',
    full_name='tensorflow.muzero.RemoteEnvironment.Initialization',
    index=0,
    containing_service=None,
    input_type=_INITIALIZATIONREQUEST,
    output_type=_INITIALIZATIONRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Step',
    full_name='tensorflow.muzero.RemoteEnvironment.Step',
    index=1,
    containing_service=None,
    input_type=_ACTIONREQUEST,
    output_type=_ACTIONRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_REMOTEENVIRONMENT)

DESCRIPTOR.services_by_name['RemoteEnvironment'] = _REMOTEENVIRONMENT

# @@protoc_insertion_point(module_scope)