import logging
from ikomia.core import task
from ikomia.utils.tests import run_for_test


logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info(f"===== Test::{t.name} =====")
    logger.info("----- Use default parameters")
    return run_for_test(t)
