import logging
from io import StringIO

import pytest

from app.utils import logger


@pytest.fixture
def log_capture():
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    logger.logger.addHandler(handler)
    yield log_stream
    logger.logger.removeHandler(handler)
    log_stream.close()


def test_log_with_capture(log_capture):
    logger.log('This is a debug message', level='DEBUG')
    logger.log('This is an info message', level='INFO')
    logger.log('This is a warning message', level='WARNING')
    logger.log('This is an error message', level='ERROR')
    logger.log('This is a critical message', level='CRITICAL')

    log_capture.seek(0)
    log_output = log_capture.read()

    assert 'This is a debug message' in log_output
    assert 'This is an info message' in log_output
    assert 'This is a warning message' in log_output
    assert 'This is an error message' in log_output
    assert 'This is a critical message' in log_output

    with pytest.raises(ValueError, match='Log level error! Unknown log level'):
        logger.log('This is an invalid level message', level='INVALID')


def test_log_decorator(log_capture):
    @logger.log_decorator('This is a test message', level='INFO')
    def test_function():
        return 'Test function executed'

    result = test_function()
    assert result == 'Test function executed'

    log_capture.seek(0)
    log_output = log_capture.read()

    assert 'This is a test message' in log_output


def test_timer_decorator(log_capture):
    @logger.timer
    def test_function():
        return 'Test function executed'

    result = test_function()
    assert result == 'Test function executed'

    log_capture.seek(0)
    log_output = log_capture.read()

    for word in ['Function', 'took', 'seconds']:
        assert word in log_output
