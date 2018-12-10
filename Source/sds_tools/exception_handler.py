"""
ML services backend basic exception handler
"""

import sys

from MLLogger import BaseMLLogger
from general_helper import logging_exception_message
from mass_transit.MTMessageProcessor import PurePublisher


class MLExceptionHandler(object):
    """
    Class to decorate some function with exception handler, logger,
    report message publisher and fail message publisher
    """

    def __init__(
            self, logger=None, report_publisher=None,
            fail_publisher=None, fail_message_constructor=None
    ):
        """
        Initialize decrator object

        :param logger: logger object (if given)
        :param report_publisher: mt_constant which expalain report exchange
        :param fail_publisher: mt_constant which expalain fail exchange
        :param fail_message_constructor: method for make fail message
        """

        if not logger:
            self.logger = generate_default_logger()
        else:
            self.logger = logger
        self.report_publisher = report_publisher
        self.fail_publisher = fail_publisher
        self.fail_message_constructor = fail_message_constructor

    def __call__(self, function):
        """
        Method which define how to wrap callback function
        with exception handler and logging all exceptions.

        :param function: callback function
        :return: wrapped with exception handler callback function
        """

        def function_wrapper(body):
            """
            Method which contain try/except blocks for function.
            If have not exceptions send report message to RabbitMQ queue
            defined in report_publisher MT constant
            If run except block logging traceback message to RabbitMQ queue
            defined in fail_publisher MT constant

            :param body: body of RabbitMQ message
            :return: function with exception handler
            """

            report_event_message = None
            report_failed_message = None

            try:
                # try to execute function code with arguments
                report_event_message = function(body)

            except Exception:
                # get all system info about exception
                exception_type, exception, traceback_text = sys.exc_info()
                # make error traceback message
                exception_message = 'Internal server error.'
                # if specified exception
                if exception_type.__name__ == Exception.__name__:
                    exception_message += ' {}'.format(exception)
                else:
                    exception_message += 'Unspecified server error.'

                # log error traceback
                logging_exception_message(self.logger)

                # make failed message for publish in rabbitmq exchange
                report_failed_message = self.fail_message_constructor(
                    exception_message, body)

            if report_event_message and self.report_publisher:
                # publish report message to rabbitmq
                report_message_publisher = PurePublisher(self.report_publisher)
                report_message_publisher.publish(report_event_message)
            elif report_event_message:
                self.logger.info(
                    'Have not report publisher, but callback is work'
                )

            if report_failed_message and self.fail_publisher:
                # publish failed message to rabbitmq
                fail_message_publisher = PurePublisher(self.fail_publisher)
                fail_message_publisher.publish(report_failed_message)
            elif report_failed_message:
                self.logger.info(
                    'Have not fail publisher, callback throw exception'
                )

        return function_wrapper


def generate_default_logger():
    """
    Method which generate default logger if logger object not given in argument

    :return: default logger object
    """

    return BaseMLLogger(
        log_name='exception_handler', log_file_name='sds-ml-exception-handler'
    )
