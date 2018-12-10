"""
Module which define basic ml logger. Should use it all times when need logging.
"""
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler


class BaseMLLogger(logging.Logger):
    """
    Class for logging some ml events
    """

    def __init__(self, log_name=None, log_file_name=None,
                 logging_level=logging.INFO):
        """
        Initialise ml logger object

        :param log_name: name of logger
        :param log_file_name: pattern name of file to logging
        :param logging_level: basic logging level
        :type log_name: str
        :type log_file_name: str
        :type logging_level: int
        """

        super().__init__(name=log_name, level=logging_level)
        set_up_base_logger()
        self.logging_level = logging_level
        self.setLevel(self.logging_level)

        # set ml formatter
        self.formatter = make_formatter()
        # add log file handler if file name exist
        self.add_rotating_handler(log_file_name)
        # add console log handler
        self.add_stream_handler()

    def add_rotating_handler(self, log_file_name):
        """
        Method for create basic logger for ML project.
        Return logger with given name which save logs to file (if given).
        Log folder should be set on environment variable

        :param log_file_name: filename to write logs
        :type log_file_name: str
        :return: prepared logger object
        """

        # set log file
        if log_file_name:
            rotation_handler = self.make_rotation_handler(log_file_name)
            self.addHandler(rotation_handler)

    def add_stream_handler(self):
        """
        Method which create and add stream handler to logger
        To see logs messages in console
        """

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(self.logging_level)

        # add formatter to handler
        stream_handler.setFormatter(self.formatter)

        # add handler to logger object
        self.addHandler(stream_handler)

    def make_rotation_handler(self, log_file_name):
        """
        Method which make rotation file handler for ml logger if filename exist

        :param log_file_name: name of file with logs
        :type log_file_name: str
        :return: prepared rotation file handler
        """

        log_folder = os.environ['OSDR_LOG_FOLDER']

        # make log dir if it bit exist
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        # add date to log file name
        log_file_name += '-{}.log'.format(time.strftime('%Y%m%d'))
        # add file handler to logger
        rot_handler = RotatingFileHandler(
            os.path.join(log_folder, log_file_name),
            maxBytes=10000, backupCount=5
        )
        rot_handler.setLevel(self.logging_level)
        rot_handler.setFormatter(self.formatter)

        return rot_handler


class MLFormatter(logging.Formatter):
    """
    Create formatter for ML logger
    """

    # set datetime converter to generate UTC (GMT +0) time
    converter = time.gmtime

    def formatTime(self, record, datefmt=None):
        """
        Redefine parent time formatter to use ML datetime logs format, UTC time

        :param record:
        :param datefmt:
        :return:
        """

        # get UTC (GMT +0) datetime object
        utc_time = self.converter(record.created)
        # if used custom logs datetime formatting
        if datefmt:
            date_time_string = time.strftime(datefmt, utc_time)
        # if used default ML logs datetime formatting
        else:
            date_time_string = time.strftime('%Y-%m-%dT%H:%M:%S', utc_time)

        return date_time_string


def make_formatter():
    """
    Method to generate ML logs formatter, based on environment variables,
    speciallyML-formatted UTC time (GMT +0).

    :return: ML logs formatter
    """

    # define app version if it come from environment variables, else empty
    app_version = ''
    if 'APP_VERSION' in os.environ:
        app_version = os.environ['APP_VERSION']

    # define machine name if it come from environment variables, else empty
    machine_name = ''
    if 'MACHINE_NAME' in os.environ:
        machine_name = os.environ['MACHINE_NAME']

    return MLFormatter(
            fmt='%(asctime)s [%(levelname)s] {}{}%(message)s'.format(
                app_version, machine_name)
        )


def set_up_base_logger():
    """
    Method to set up base logger. It set up log levels names and base log level
    """

    # change default logs levels names
    logging.addLevelName(50, 'FATAL')
    logging.addLevelName(40, 'ERROR')
    logging.addLevelName(30, 'WARN')
    logging.addLevelName(20, 'INFO')
    logging.addLevelName(10, 'DEBUG')

    # change base logging level to INFO
    logging.basicConfig(level=logging.INFO)
