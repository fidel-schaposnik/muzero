# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: muzero/protos/environment.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow.core.framework import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='muzero/protos/environment.proto',
  package='tensorflow.muprover',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1fmuzero/protos/environment.proto\x12\x13tensorflow.muprover\x1a&tensorflow/core/framework/tensor.proto\"L\n\x05State\x12,\n\x0bobservation\x18\x01 \x01(\x0b\x32\x17.tensorflow.TensorProto\x12\x15\n\rlegal_actions\x18\x02 \x03(\x05\"\xbc\x01\n\x15InitializationRequest\x12\x65\n\x16\x65nvironment_parameters\x18\x01 \x03(\x0b\x32\x45.tensorflow.muprover.InitializationRequest.EnvironmentParametersEntry\x1a<\n\x1a\x45nvironmentParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01\"A\n\x16InitializationResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x16\n\x0e\x65nvironment_id\x18\x02 \x01(\t\"-\n\x13\x46inalizationRequest\x12\x16\n\x0e\x65nvironment_id\x18\x01 \x01(\t\"\'\n\x14\x46inalizationResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\"&\n\x0cResetRequest\x12\x16\n\x0e\x65nvironment_id\x18\x01 \x01(\t\"K\n\rResetResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12)\n\x05state\x18\x02 \x01(\x0b\x32\x1a.tensorflow.muprover.State\"5\n\x0bStepRequest\x12\x16\n\x0e\x65nvironment_id\x18\x01 \x01(\t\x12\x0e\n\x06\x61\x63tion\x18\x02 \x01(\x05\"\xd0\x01\n\x0cStepResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12)\n\x05state\x18\x02 \x01(\x0b\x32\x1a.tensorflow.muprover.State\x12\x0e\n\x06reward\x18\x03 \x01(\x02\x12\x0c\n\x04\x64one\x18\x04 \x01(\x08\x12\x39\n\x04info\x18\x05 \x03(\x0b\x32+.tensorflow.muprover.StepResponse.InfoEntry\x1a+\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x0c:\x02\x38\x01\x32\x88\x03\n\x11RemoteEnvironment\x12k\n\x0eInitialization\x12*.tensorflow.muprover.InitializationRequest\x1a+.tensorflow.muprover.InitializationResponse\"\x00\x12\x65\n\x0c\x46inalization\x12(.tensorflow.muprover.FinalizationRequest\x1a).tensorflow.muprover.FinalizationResponse\"\x00\x12M\n\x04Step\x12 .tensorflow.muprover.StepRequest\x1a!.tensorflow.muprover.StepResponse\"\x00\x12P\n\x05Reset\x12!.tensorflow.muprover.ResetRequest\x1a\".tensorflow.muprover.ResetResponse\"\x00\x62\x06proto3'
  ,
  dependencies=[tensorflow_dot_core_dot_framework_dot_tensor__pb2.DESCRIPTOR,])




_STATE = _descriptor.Descriptor(
  name='State',
  full_name='tensorflow.muprover.State',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='observation', full_name='tensorflow.muprover.State.observation', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='legal_actions', full_name='tensorflow.muprover.State.legal_actions', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=96,
  serialized_end=172,
)


_INITIALIZATIONREQUEST_ENVIRONMENTPARAMETERSENTRY = _descriptor.Descriptor(
  name='EnvironmentParametersEntry',
  full_name='tensorflow.muprover.InitializationRequest.EnvironmentParametersEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.muprover.InitializationRequest.EnvironmentParametersEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.muprover.InitializationRequest.EnvironmentParametersEntry.value', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=303,
  serialized_end=363,
)

_INITIALIZATIONREQUEST = _descriptor.Descriptor(
  name='InitializationRequest',
  full_name='tensorflow.muprover.InitializationRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='environment_parameters', full_name='tensorflow.muprover.InitializationRequest.environment_parameters', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=175,
  serialized_end=363,
)


_INITIALIZATIONRESPONSE = _descriptor.Descriptor(
  name='InitializationResponse',
  full_name='tensorflow.muprover.InitializationResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='tensorflow.muprover.InitializationResponse.success', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='environment_id', full_name='tensorflow.muprover.InitializationResponse.environment_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=365,
  serialized_end=430,
)


_FINALIZATIONREQUEST = _descriptor.Descriptor(
  name='FinalizationRequest',
  full_name='tensorflow.muprover.FinalizationRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='environment_id', full_name='tensorflow.muprover.FinalizationRequest.environment_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=432,
  serialized_end=477,
)


_FINALIZATIONRESPONSE = _descriptor.Descriptor(
  name='FinalizationResponse',
  full_name='tensorflow.muprover.FinalizationResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='tensorflow.muprover.FinalizationResponse.success', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=479,
  serialized_end=518,
)


_RESETREQUEST = _descriptor.Descriptor(
  name='ResetRequest',
  full_name='tensorflow.muprover.ResetRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='environment_id', full_name='tensorflow.muprover.ResetRequest.environment_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=520,
  serialized_end=558,
)


_RESETRESPONSE = _descriptor.Descriptor(
  name='ResetResponse',
  full_name='tensorflow.muprover.ResetResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='tensorflow.muprover.ResetResponse.success', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='state', full_name='tensorflow.muprover.ResetResponse.state', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=560,
  serialized_end=635,
)


_STEPREQUEST = _descriptor.Descriptor(
  name='StepRequest',
  full_name='tensorflow.muprover.StepRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='environment_id', full_name='tensorflow.muprover.StepRequest.environment_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='action', full_name='tensorflow.muprover.StepRequest.action', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=637,
  serialized_end=690,
)


_STEPRESPONSE_INFOENTRY = _descriptor.Descriptor(
  name='InfoEntry',
  full_name='tensorflow.muprover.StepResponse.InfoEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorflow.muprover.StepResponse.InfoEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorflow.muprover.StepResponse.InfoEntry.value', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=858,
  serialized_end=901,
)

_STEPRESPONSE = _descriptor.Descriptor(
  name='StepResponse',
  full_name='tensorflow.muprover.StepResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='tensorflow.muprover.StepResponse.success', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='state', full_name='tensorflow.muprover.StepResponse.state', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='reward', full_name='tensorflow.muprover.StepResponse.reward', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='done', full_name='tensorflow.muprover.StepResponse.done', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='info', full_name='tensorflow.muprover.StepResponse.info', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_STEPRESPONSE_INFOENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=693,
  serialized_end=901,
)

_STATE.fields_by_name['observation'].message_type = tensorflow_dot_core_dot_framework_dot_tensor__pb2._TENSORPROTO
_INITIALIZATIONREQUEST_ENVIRONMENTPARAMETERSENTRY.containing_type = _INITIALIZATIONREQUEST
_INITIALIZATIONREQUEST.fields_by_name['environment_parameters'].message_type = _INITIALIZATIONREQUEST_ENVIRONMENTPARAMETERSENTRY
_RESETRESPONSE.fields_by_name['state'].message_type = _STATE
_STEPRESPONSE_INFOENTRY.containing_type = _STEPRESPONSE
_STEPRESPONSE.fields_by_name['state'].message_type = _STATE
_STEPRESPONSE.fields_by_name['info'].message_type = _STEPRESPONSE_INFOENTRY
DESCRIPTOR.message_types_by_name['State'] = _STATE
DESCRIPTOR.message_types_by_name['InitializationRequest'] = _INITIALIZATIONREQUEST
DESCRIPTOR.message_types_by_name['InitializationResponse'] = _INITIALIZATIONRESPONSE
DESCRIPTOR.message_types_by_name['FinalizationRequest'] = _FINALIZATIONREQUEST
DESCRIPTOR.message_types_by_name['FinalizationResponse'] = _FINALIZATIONRESPONSE
DESCRIPTOR.message_types_by_name['ResetRequest'] = _RESETREQUEST
DESCRIPTOR.message_types_by_name['ResetResponse'] = _RESETRESPONSE
DESCRIPTOR.message_types_by_name['StepRequest'] = _STEPREQUEST
DESCRIPTOR.message_types_by_name['StepResponse'] = _STEPRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

State = _reflection.GeneratedProtocolMessageType('State', (_message.Message,), {
  'DESCRIPTOR' : _STATE,
  '__module__' : 'muzero.protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muprover.State)
  })
_sym_db.RegisterMessage(State)

InitializationRequest = _reflection.GeneratedProtocolMessageType('InitializationRequest', (_message.Message,), {

  'EnvironmentParametersEntry' : _reflection.GeneratedProtocolMessageType('EnvironmentParametersEntry', (_message.Message,), {
    'DESCRIPTOR' : _INITIALIZATIONREQUEST_ENVIRONMENTPARAMETERSENTRY,
    '__module__' : 'muzero.protos.environment_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.muprover.InitializationRequest.EnvironmentParametersEntry)
    })
  ,
  'DESCRIPTOR' : _INITIALIZATIONREQUEST,
  '__module__' : 'muzero.protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muprover.InitializationRequest)
  })
_sym_db.RegisterMessage(InitializationRequest)
_sym_db.RegisterMessage(InitializationRequest.EnvironmentParametersEntry)

InitializationResponse = _reflection.GeneratedProtocolMessageType('InitializationResponse', (_message.Message,), {
  'DESCRIPTOR' : _INITIALIZATIONRESPONSE,
  '__module__' : 'muzero.protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muprover.InitializationResponse)
  })
_sym_db.RegisterMessage(InitializationResponse)

FinalizationRequest = _reflection.GeneratedProtocolMessageType('FinalizationRequest', (_message.Message,), {
  'DESCRIPTOR' : _FINALIZATIONREQUEST,
  '__module__' : 'muzero.protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muprover.FinalizationRequest)
  })
_sym_db.RegisterMessage(FinalizationRequest)

FinalizationResponse = _reflection.GeneratedProtocolMessageType('FinalizationResponse', (_message.Message,), {
  'DESCRIPTOR' : _FINALIZATIONRESPONSE,
  '__module__' : 'muzero.protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muprover.FinalizationResponse)
  })
_sym_db.RegisterMessage(FinalizationResponse)

ResetRequest = _reflection.GeneratedProtocolMessageType('ResetRequest', (_message.Message,), {
  'DESCRIPTOR' : _RESETREQUEST,
  '__module__' : 'muzero.protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muprover.ResetRequest)
  })
_sym_db.RegisterMessage(ResetRequest)

ResetResponse = _reflection.GeneratedProtocolMessageType('ResetResponse', (_message.Message,), {
  'DESCRIPTOR' : _RESETRESPONSE,
  '__module__' : 'muzero.protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muprover.ResetResponse)
  })
_sym_db.RegisterMessage(ResetResponse)

StepRequest = _reflection.GeneratedProtocolMessageType('StepRequest', (_message.Message,), {
  'DESCRIPTOR' : _STEPREQUEST,
  '__module__' : 'muzero.protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muprover.StepRequest)
  })
_sym_db.RegisterMessage(StepRequest)

StepResponse = _reflection.GeneratedProtocolMessageType('StepResponse', (_message.Message,), {

  'InfoEntry' : _reflection.GeneratedProtocolMessageType('InfoEntry', (_message.Message,), {
    'DESCRIPTOR' : _STEPRESPONSE_INFOENTRY,
    '__module__' : 'muzero.protos.environment_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.muprover.StepResponse.InfoEntry)
    })
  ,
  'DESCRIPTOR' : _STEPRESPONSE,
  '__module__' : 'muzero.protos.environment_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.muprover.StepResponse)
  })
_sym_db.RegisterMessage(StepResponse)
_sym_db.RegisterMessage(StepResponse.InfoEntry)


_INITIALIZATIONREQUEST_ENVIRONMENTPARAMETERSENTRY._options = None
_STEPRESPONSE_INFOENTRY._options = None

_REMOTEENVIRONMENT = _descriptor.ServiceDescriptor(
  name='RemoteEnvironment',
  full_name='tensorflow.muprover.RemoteEnvironment',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=904,
  serialized_end=1296,
  methods=[
  _descriptor.MethodDescriptor(
    name='Initialization',
    full_name='tensorflow.muprover.RemoteEnvironment.Initialization',
    index=0,
    containing_service=None,
    input_type=_INITIALIZATIONREQUEST,
    output_type=_INITIALIZATIONRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Finalization',
    full_name='tensorflow.muprover.RemoteEnvironment.Finalization',
    index=1,
    containing_service=None,
    input_type=_FINALIZATIONREQUEST,
    output_type=_FINALIZATIONRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Step',
    full_name='tensorflow.muprover.RemoteEnvironment.Step',
    index=2,
    containing_service=None,
    input_type=_STEPREQUEST,
    output_type=_STEPRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Reset',
    full_name='tensorflow.muprover.RemoteEnvironment.Reset',
    index=3,
    containing_service=None,
    input_type=_RESETREQUEST,
    output_type=_RESETRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_REMOTEENVIRONMENT)

DESCRIPTOR.services_by_name['RemoteEnvironment'] = _REMOTEENVIRONMENT

# @@protoc_insertion_point(module_scope)
