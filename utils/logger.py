# SPDX-FileCopyrightText: 2026 Devine Ngouloubi <exauce-devine.ngouloubi@unicaen.fr>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
 @file logger.py
 @brief Centralised logging configuration for the application.
 @details Provides tools to initialise logging and a Mixin for the classes.
"""

import logging
import sys
from typing import Optional

def setup_logging(level=logging.INFO, log_file: Optional[str] = None):
    """!
    @brief Configures the global logging for the application.
    @details Initialises the handlers for standard output and optionally for a file.
    Also defines the message format and adjusts the level of third-party libraries.
    @param level int The logging severity level (e.g. logging.INFO, logging.DEBUG).
    @param log_file Optional[str] Optional path to a log file.
    @return None
    """

    # Custom formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handlers
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Apply the formatter to the StreamHandler
    handlers[0].setFormatter(formatter)

    # Main configuration
    # Force the config to override any potential default configs
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )

    # Reduce the verbosity of some noisy third-party libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

class LoggingMixin:
    """!
    @class LoggingMixin
    @brief Mixin to easily add a logger instance to classes.
    """

    @property
    def logger(self):
        """!
        @brief Gets or creates a logger specific to the class.
        @return logging.Logger A logging.Logger instance named after the class.
        """
        name = '.'.join([__name__, self.__class__.__name__])
        return logging.getLogger(name)
