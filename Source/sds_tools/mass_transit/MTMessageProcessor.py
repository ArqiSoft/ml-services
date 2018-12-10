"""
Module for send and receive Mass Transit messages with Python.
"""
from mass_transit.MTMessageProcessorHelper import (
    make_mt_exchange_name, make_rabbit_connection, get_channel_from_connection,
    declare_exchange, dict_to_masstransit_message, make_queue_name,
    get_dict_from_body, LOGGER, get_schema_types_from_file, add_default_values,
    get_namespace, get_hostname, get_event, get_command
)


class PurePublisher(object):
    """
    Class for simplifying publish message from Python to MassTransit (MT)
    """

    def __init__(self, mt_constant, channel=None):
        """
        Method for define all needed for publishing variables.

        :param mt_constant: dict with emulation of MT interfaces
        :param channel: pika connection channel object. to which publish
        """

        self.mt_namespace = get_namespace(mt_constant)
        self.hostname = get_hostname(mt_constant)
        self.connection = make_rabbit_connection(self.hostname)
        # use Publisher for send events only
        self.channel = channel if channel else self.connection.channel()
        self.mt_endpoint = get_event(mt_constant)
        self.publish_schema_types = get_schema_types_from_file(
            mt_constant['event_json_file'])
        self.mt_exchange_name = make_mt_exchange_name(
            self.mt_namespace, self.mt_endpoint
        )
        self.message_type = self.make_message_type()

    def publish(self, message):
        """
        Method for send message to MT listener exchange.
        Declare needed exchange, then add message type to bo body and
        convert they both to MT format, send converted message to MT exchange,
        finally write log and close connection with RabbitMQ

        :param message: body of MT 'message'. MUST be dict
        """

        self.publishing_message_to_exchange(message)

        # close connection to interrupt infinities publishing
        self.connection.close()

    def publishing_message_to_exchange(self, message):
        """
        Method which publish message to RabbitMQ queue.

        :param message: dict with data which we want to publish
        :return: body of message which is sended to RabbtitMQ,
                    or None if something wrong
        """

        # declare exchange for publishing
        # correspond to exchange which MT listening
        mt_body = self.prepare_publishing_body(message)

        if mt_body:
            declare_exchange(self.channel, self.mt_exchange_name)

            # publish message to MT listening exchange
            self.channel.basic_publish(
                exchange=self.mt_exchange_name, body=mt_body, routing_key=''
            )

            LOGGER.info('[x] Sent message: {} to exchange name: {}'.format(
                mt_body, self.mt_exchange_name
            ))

        return mt_body

    def prepare_publishing_body(self, message):
        """
        Method which validate message parameters names and add message type,
        needed for MT, to body of message for RabbitMQ

        :param message: dict with data which we want to publish
        :return: prepared body of message or None if something wrong
        """

        # # if output parameters names or types wrong
        # if not parameters_is_valid(message, self.publish_schema_types):
        #     LOGGER.error('ERROR WHILE VALIDATE OUTPUT MESSAGE')
        #     return None

        # add message_type to 'message' and transform it both to MT format
        mt_body = dict_to_masstransit_message(self.message_type, message)

        return mt_body

    def make_event_consumer(self, mt_constant):
        """
        Method which make Consumer for Publisher. If Publisher used
        for publish command messages, we should listen for event messages

        :param mt_constant: dict with emulation of MT interfaces
        :return: Consumer object
        """

        return PureConsumer(mt_constant, channel=self.channel)

    def make_message_type(self):
        """
        Method which generate messageType value for MT listener.
        Better see MT convention about this variable

        :return: needed messageType value as string
        """

        return 'urn:message:{}'.format(self.mt_exchange_name)


class MTPublisher(PurePublisher):
    """
    Class for simplifying publish message from Python to MassTransit (MT) and
    consuming for events messages
    """

    def __init__(self, mt_constant):
        """
        Method for define all needed for publishing variables.

        :param mt_constant: dict with emulation of MT interfaces
        """

        super().__init__(mt_constant)

        # use Publisher for send commands and receive events
        self.mt_namespace = get_namespace(mt_constant, command=True)
        self.mt_endpoint = get_command(mt_constant)
        self.channel = get_channel_from_connection(self.connection)
        self.publish_schema_types = get_schema_types_from_file(
            mt_constant['command_json_file'])
        self.event_consumer = self.make_event_consumer(mt_constant)
        self.mt_exchange_name = make_mt_exchange_name(
            self.mt_namespace, self.mt_endpoint
        )
        self.message_type = self.make_message_type()

    def publish(self, message_body):
        """
        Method for publish message to MT listener exchange.
        Declare needed exchange, then add message type to bo body and
        convert they both to MT format, send converted message to MT exchange,
        finally write log and IF have published some not empty command,
        start listening for event message,
        after receive event message close connection with RabbitMQ

        :param message_body: body of MT 'message'. MUST be dict
        """

        self.event_consumer.set_consumer()
        sent_message = self.publishing_message_to_exchange(message_body)

        # if Publisher used for send command we should consume events
        if sent_message:
            self.event_consumer.start_consuming()

        # close connection to interrupt infinities publishing
        self.connection.close()


class PureConsumer(object):
    """
    Class for simplifying consume message from MassTransit (MT) to Python
    """

    def __init__(
            self, mt_constant, channel=None, infinite_consuming=False,
            prefetch_count=1
    ):
        """
        Method for define all needed for consuming variables.

        :param mt_constant: dict with emulation of MT interfaces
        :param channel: pika connection channel object. from what consume
        """

        self.infinite_consuming = infinite_consuming
        self.hostname = get_hostname(mt_constant)
        self.mt_namespace = get_namespace(mt_constant)
        self.connection = make_rabbit_connection(self.hostname)
        self.need_upper = mt_constant['need_upper']
        # use Consumer for receive events only
        self.channel = channel if channel else self.connection.channel()
        self.mt_endpoint = get_event(mt_constant)
        self.consume_schema_types = get_schema_types_from_file(
            mt_constant['event_json_file'])
        self.callback_function = self.add_event_to_callback_function(
            mt_constant['event_callback']
        )
        self.queue_name = make_queue_name(
            self.mt_namespace, mt_constant['publisher_queue_name']
        )
        self.mt_exchange_name = make_mt_exchange_name(
            self.mt_namespace, self.mt_endpoint
        )
        # prefetch count, max number of threads
        self.prefetch_count = prefetch_count

    def add_event_to_callback_function(self, callback_function):
        """
        Method which add event sender wrapper to callback function. We need it
        because MT sender listening for operation result from python consumer

        :param callback_function: MT command handler function
        :return: callback_function wrapped with event sender
        """

        def callback_wrapper(channel, method, properties, body):
            """
            Method which execute user's callback function, then stop consuming

            :param channel: default pika callback function parameter
            :param method: default pika callback function parameter
            :param properties: default pika callback function parameter
            :param body: default pika callback function parameter
            """

            # get serialized 'message' from RabbitMQ's message body
            message_dict = self.get_message_from_body(body)
            LOGGER.info('[x] Received message: {}'.format(body.decode()))

            # add None values
            message_dict = add_default_values(
                message_dict, self.consume_schema_types
            )
            callback_function(message_dict)
            channel.basic_ack(delivery_tag=method.delivery_tag)

            # stop consuming if Consumer object execute from Publisher,
            # if Consumer had listening for event, not for command
            if not self.infinite_consuming:
                self.channel.stop_consuming()

        return callback_wrapper

    def get_message_from_body(self, body):
        """
        Method for get valid message from body of RabbitMQ message

        :param body: body of RabbitMQ message
        :return: valid message or None if message not valid
        """

        # take 'message' from MT body as dict WITH or WITHOUT
        # force capitalize first chars of all parameters names
        message_dict = get_dict_from_body(body, need_upper=self.need_upper)

        # # validation of all input 'message' parameters
        # # check input 'message' parameters for correctness
        # if not parameters_is_valid(message_dict, self.consume_schema_types):
        #     LOGGER.error('ERROR WHILE VALIDATE INPUT MESSAGE')
        #     return None

        return message_dict

    def start_consuming(self):
        """
        Method which initiate consuming process.
        Set basic consumer for Consumer object, write log about it, and
        start infinite consuming process (use CTRL+C to interrupt consuming)
        """

        # set basic consumer for Consumer object
        self.set_consumer()

        LOGGER.info(
            '[*] Waiting for data from exchange: {}, queue: {}'.format(
                self.mt_exchange_name, self.queue_name)
        )

        # start consume process
        self.channel.start_consuming()

    def set_consumer(self):
        """
        Method which execute operations needed for consuming by Consumer object
        Such as exchange declaring, queue name making, queue to channel binding
        and defining basic consumer.
        All operations look like as operations in RabbitMQ tutorial
        """

        # set maximum reading messages count
        self.channel.basic_qos(prefetch_count=self.prefetch_count)
        # declare exchange for listening
        # correspond to exchange which MT publishing
        declare_exchange(self.channel, self.mt_exchange_name)

        self.channel.queue_declare(queue=self.queue_name, durable=True)

        # bind queue to exhchange
        self.channel.queue_bind(
            exchange=self.mt_exchange_name, queue=self.queue_name)

        # declare consumer
        self.channel.basic_consume(
            self.callback_function, queue=self.queue_name)

    def make_event_publisher(self, mt_constant):
        """
        Method for make linked event publisher,
        which would linked to MT event listener
        """

        # we use consumer channel, because of MT publisher waiting for
        # event message on that channel
        return PurePublisher(mt_constant, channel=self.channel)


class MTConsumer(PureConsumer):
    """
    Class for simplifying consume message from MassTransit (MT) to Python, and
    send back event message
    """

    def __init__(self, mt_constant, infinite_consuming=False,
                 prefetch_count=1):
        """
        Method for define all needed for consuming variables.

        :param mt_constant: dict with emulation of MT interfaces
        """

        super().__init__(
            mt_constant, infinite_consuming=infinite_consuming,
            prefetch_count=prefetch_count
        )

        # use Consumer for receive commands and send events
        self.mt_endpoint = get_command(mt_constant)
        self.channel = get_channel_from_connection(self.connection)
        self.consume_schema_types = get_schema_types_from_file(
            mt_constant['command_json_file'])
        self.event_publisher = self.make_event_publisher(mt_constant)
        self.callback_function = self.add_event_to_callback_function(
            mt_constant['command_callback']
        )
        self.queue_name = make_queue_name(
            self.mt_namespace, mt_constant['consumer_queue_name']
        )
        self.mt_exchange_name = make_mt_exchange_name(
            self.mt_namespace, self.mt_endpoint
        )

    def add_event_to_callback_function(self, callback_function):
        """
        Method which add event sender wrapper to callback function. We need it
        because MT sender listening for operation result from python consumer

        :param callback_function: MT command handler function
        :return: callback_function wrapped with event sender
        """

        def callback_wrapper(channel, method, properties, body):
            """
            Method which execute user's callback function, then send back
            operation result (aka event) to MT sender

            :param channel: default pika callback function parameter
            :param method: default pika callback function parameter
            :param properties: default pika callback function parameter
            :param body: default pika callback function parameter
            """

            # get serialized 'message' from RabbitMQ's message body
            message_dict = self.get_message_from_body(body)
            LOGGER.info('[x] Received message: {}'.format(body.decode()))
            message_dict = add_default_values(
                message_dict, self.consume_schema_types
            )
            event_message = callback_function(message_dict)
            channel.basic_ack(delivery_tag=method.delivery_tag)

            # send event back to MT consumer
            self.event_publisher.publish(event_message)

        return callback_wrapper
