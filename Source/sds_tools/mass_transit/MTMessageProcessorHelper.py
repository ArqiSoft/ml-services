"""
General methods for MT constants and MT-emulation Python classes
"""

import json
import uuid

import pika

from MLLogger import BaseMLLogger

LOGGER = BaseMLLogger(
    log_name='mt_library_logger',log_file_name='sds-mt-library')

# json to Python types mapping table
# key is json type, value is Python type
TYPES_TABLE = {
    'string': str,
    'integer': int,
    'decimal': float,
    'boolean': bool,
    'array': list,
    'Guid': uuid.UUID,
    'object': dict
}


class Guid(str):
    def __init__(self, string_value):
        try:
            uuid.UUID(string_value)
        except ValueError:
            super().__init__()


def dict_to_masstransit_message(message_type, message_body_dict):
    """
    Method for convert dict with 'message' body and message_type value
    to MT readable format

    :param message_type: message_type value, need for MT body
    :param message_body_dict: dict with data of 'message' for MT
    :return: MT body, ready for transfer
    """

    # fix formatting and tranform dict to bytes
    masstransit_message = json.dumps({
        'messageType': [message_type],
        'message': message_body_dict,
    })

    return masstransit_message


def masstransit_message_to_dict(masstransit_message):
    """
    Method for convert MT message to dict

    :param masstransit_message: MT message (json in bytes) we want to convert
    :return: dict with all data from MT message
    """

    return json.loads(masstransit_message.decode('utf-8'))


def make_rabbit_connection(hostname):
    """
    Method for create connection to RabbitMQ by hostname.

    :return: python pika BlockingConnection object
    """
    url_parameters = pika.URLParameters(hostname)

    return pika.BlockingConnection(pika.ConnectionParameters(
        host=url_parameters.host, port=url_parameters.port,
        virtual_host=url_parameters.virtual_host, heartbeat_interval=0
    ))


def get_channel_from_connection(pika_connection_object):
    """
    Method which return channel object from pika connection

    :return: channel object of pika.connection object
    """

    return pika_connection_object.channel()


def get_dict_from_body(body, need_upper=False):
    """
    Method which return 'message' from MT body as dict.
    In 'message' all dict parameters names with lower first char, so
    we need change first char to upper case to make 'message' parameter name
    correspond to contract parameter name

    :param body: message as bytes from RabbitMQ queue
    :param need_upper:
                True if need to force capitalize first char of parameters names
                False if leaev parameters names 'as is'
    :return: body 'message' as dict with upper first chars of parameters names
    """

    # get 'message'
    not_upper_message_as_dict = json.loads(body.decode('utf-8'))['message']

    # return 'message' as dict with parameters names 'as is'
    if not need_upper:
        return not_upper_message_as_dict

    # prepare to return 'message' as dict with upper case first chars
    # for parameters names
    message_as_dict = {}
    for key, value in not_upper_message_as_dict.items():
        message_as_dict[key[0].upper() + key[1:]] = value

    return message_as_dict


def parameters_is_valid(message, schema_types):
    """
    Method for validate messages's parameters names and types

    :param message: message which contain all parameters
    :param schema_types: dict with parameters names as key and type as value
    :return: True if message's parameters is valid, False if not valid
    """

    required_properties = schema_types['required']
    # only required parameters will be validated
    if required_properties:
        json_schema_as_dict = dict()
        for property_name, property_type in schema_types['properties'].items():
            # check if property name in required list
            # set property as required
            if property_name in required_properties:
                json_schema_as_dict[property_name] = property_type

    # if json file have not "required" field, so all properties are required
    else:
        json_schema_as_dict = schema_types['properties']

    # convert message from json to python types
    converted_message = convert_message_types(message, schema_types)

    # return message validation result
    return validate_data_with_schema(json_schema_as_dict, converted_message)


def get_schema_types_from_file(file_path):
    """
    Method for get dict with property name as key, property type as value
    from json schema file

    :param file_path: path to json schema file
    :return: dict with property name as key and property type as value
    """

    # get json data as dict from json schema file
    json_data = get_json_from_file(file_path)

    # convert schema types to python types
    schema_types = convert_schema_types(json_data)
    # get default values of properies from schema
    default_values = get_default_values(json_data)
    schema_types.update({'defaults': default_values})

    return schema_types


def get_default_values(json_data):
    """
    Method to make default values dict for properties in json schema.
    Set default value as None if property havent 'default' key in schema

    :param json_data: json schema with properties as dict with 'default'
                    keys or without it
    :return: dict with default values for all properties in given schema
    """

    default_properties_values = dict()
    # make defaults for all properties
    for property_name in json_data['properties']:
        # if schema have 'default' value for property
        if 'default' in json_data['properties'][property_name].keys():
            default_value = json_data['properties'][property_name]['default']
        # use None as default default value
        else:
            default_value = None

        default_properties_values[property_name] = default_value

    return default_properties_values


def get_json_from_file(file_path):
    """
    Get json data from file wich placed on file_path as dict

    :param file_path: path to json file
    :return: json data as dict from file
    """

    # get json bytes from json file
    json_file = open(file_path, 'r')
    # convert json bytes to python dict
    json_data_as_dict = json.load(json_file)
    # close json file
    json_file.close()
    # return json data as dict
    return json_data_as_dict


def convert_schema_types(json_data):
    """
    Method for convert types from json to python. Convert types with help of
    conversation table (TYPES_TABLE)

    :param json_data: dict with json schema types
    :return: dict of schema with python types as value and property name as key
    """

    # define initial schema dict
    schema_dict = dict()
    schema_dict['required'] = None
    schema_dict['properties'] = {}

    # fill names of required schema parameters
    if 'required' in json_data.keys():
        schema_dict['required'] = json_data['required']

    # fill properties with python property types
    for property_name, property_type in json_data['properties'].items():
        # get python type from table
        # if have special json type
        if 'format' in property_type.keys():
            python_type = TYPES_TABLE[property_type['format']]
        # if simple json type
        else:
            python_type = TYPES_TABLE[property_type['type']]

        # add property to schema with python types
        schema_dict['properties'][property_name] = python_type

    return schema_dict


def validate_data_with_schema(schema, data):
    """
    Method for validate data with given schema.
    All required by schema properties MUST be in data.
    All required properties types MUST correspond to schema.

    :param schema: schema object which we will use as template for validation
    :param data: dict with data which we want to validate
    :return: True if data correspond with schema,
            False if data not correspond with schema
    """

    data_keys = data.keys()
    not_suitable_types = []
    for key, value in schema.items():
        if key in data_keys:
            if not isinstance(data[key], value):
                not_suitable_types.append(
                    'key {} value, not type of {}'.format(key, value)
                )

    if not_suitable_types:
        error_message = 'Wrong key value: {}'.format(
            ','.join(not_suitable_types)
        )
        LOGGER.error('Error while validate schema: {}'.format(error_message))

    if not_suitable_types:
        return False

    return True


def convert_message_types(message, schema_types):
    """
    Method which try to convert message with json data to dict with python data
    which would be correspond to schema. Convert only required properties

    :param message: message which we want to convert
    :param schema_types: schema with types which we want to use for message
    :return: dict with converted properties types
    """

    # define initial message dict for python types
    message_dict = dict()
    # get all required properties names
    schema_properties_names = schema_types['properties'].keys()
    for property_name, property_value in message.items():
        # if current property required, convert it to python type
        if property_name in schema_properties_names:
            converted_property_value = convert_value_to_type(
                property_value, schema_types['properties'][property_name])
        else:
            converted_property_value = property_value

        # add converted property to dict
        message_dict[property_name] = converted_property_value

    return message_dict


def convert_value_to_type(value, type_name):
    """
    Method which try to convert any given value to any given type

    :param value: value which we want to convert
    :param type_name: type to which we want convert value
    :return: value converted to given type or given value if can not convert
    """

    # define initial converted value as value for exception case
    converted_value = value
    if isinstance(converted_value, type(None)):
        return converted_value

    try:
        # if can convert value to given type, do it
        converted_value = type_name(value)
    except ValueError:
        # write log if can not convert value to given type
        LOGGER.error('Can not convert value: {} to type {}'.format(
            value, type_name))

    return converted_value


def make_mt_exchange_name(namespace, endpoint):
    """
    Method which generate MT name which based on MT namespace
    and endpoint name. Better see MT documentation for known details

    :param namespace: MT namespace
    :param endpoint: MT endpoint name
    :return: RabbitMQ exchange name needed for pika as string
    """

    return '{}:{}'.format(namespace, endpoint)


def declare_exchange(channel, exchange_name):
    """
    Method which declare new exchange correspond to MT exchange.
    Declare new exchange for channel, return nothing

    :param channel: pika blocking channel object
    :param exchange_name: MT exchange name
    """
    channel.exchange_declare(
        exchange=exchange_name, exchange_type='fanout', durable=True
    )


def make_queue_name(mt_namespace, handler_name):
    """
    Method for declare new queue name in channel. Depends on queue "type",
    is it receive event or command.

    :param mt_namespace: string with Mass Transit namespace
    :param handler_name: string with queue time. MUST be 'command' or 'event'
    :return: new unique queue name for channel
    """

    return '{}.{}'.format(mt_namespace, handler_name)


def add_default_values(message_dict, consume_schema_types):
    """
    Method add default values for all properties in given message
    to correspond given json schema.
    See default value in schema or it would be None

    :param message_dict: properties with its values as dict
    :param consume_schema_types: json schema as dict
    :return: properties with its values as dict with filled all
            missed properties by default values from json schema
    """

    # check all properties in schema
    for property_name in consume_schema_types['properties'].keys():
        # add default value if message havent property
        if property_name not in message_dict.keys():
            default_value = consume_schema_types['defaults'][property_name]
            message_dict[property_name] = default_value

    return message_dict


def get_command(mt_constant):
    """
    Method which return command exchange name
    from json schema linked to MT constant object

    :param mt_constant: dict from mass_transit_constants
    :return: string value of command exchange name from json schema
    """

    json_as_dict = get_json_from_file(mt_constant['command_json_file'])

    if 'command' in json_as_dict:
        return json_as_dict['command']
    else:
        return json_as_dict['event']


def get_event(mt_constant):
    """
    Method which return event exchange name
    from json schema linked to MT constant object

    :param mt_constant: dict from mass_transit_constants
    :return: string value of event exchange name from json schema
    """

    json_as_dict = get_json_from_file(mt_constant['event_json_file'])

    if 'event' in json_as_dict:
        return json_as_dict['event']
    else:
        return json_as_dict['command']


def get_hostname(mt_constant):
    """
    Method which return hostname from json schema linked to MT constant object

    :param mt_constant: dict from mass_transit_constants
    :return: string value of hostname from json schema
    """

    json_as_dict = None
    # we can have only one json schema file. So use only one schema
    if 'event_json_file' in mt_constant.keys():
        json_as_dict = get_json_from_file(mt_constant['event_json_file'])
    elif 'command_json_file' in mt_constant.keys():
        json_as_dict = get_json_from_file(mt_constant['command_json_file'])

    return json_as_dict['hostname'] if json_as_dict else None


def get_namespace(mt_constant, command=False):
    """
    Method which return namespace from json schema linked to MT constant object

    :param mt_constant: dict from mass_transit_constants
    :param command: connamd or event parameter flag.
        Set it True if you get command namespace.
        Set it False if you get event namespace.
    :return: string value of namespace from json schema
    """

    # we can have only one json schema file. So use only one schema
    if command:
        json_as_dict = get_json_from_file(mt_constant['command_json_file'])
    else:
        json_as_dict = get_json_from_file(mt_constant['event_json_file'])

    return json_as_dict['namespace'] if json_as_dict else None
