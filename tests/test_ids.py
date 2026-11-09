"""Tests for `sass.ids.extract_base_id`."""

import pytest
import pytest_check as check

from sass.ids import extract_base_id


@pytest.mark.parametrize(
    ("input_id", "output_id", "keep_isomer_warts"),
    [
        ("123456", "123456", False),
        ("123456_1", "123456", False),
        ("123456_1", "123456_1", True),
        ("123456_1_2", "123456_1", False),
        ("123456_1_2", "123456_1_2", True),
        ("foo_bar", "foo_bar", False),
        ("foo_bar_1", "foo_bar", False),
        ("foo_bar_1", "foo_bar_1", True),
        ("foo_bar_1_2", "foo_bar_1", False),
        ("foo_bar_1_2", "foo_bar_1_2", True),
        # A base whose last segment is not a number has no wart to strip.
        ("t_bisamide_c_isonicotinoyl", "t_bisamide_c_isonicotinoyl", False),
        # Single-fragment ids append the connector tag after the wart.
        ("7_1[Np]", "7", False),
        ("t_bisamide_b_pentanoyl_1[U]", "t_bisamide_b_pentanoyl", False),
        # Product ids join synthons with `&`; the wart still comes off the end.
        ("12345&67890_1", "12345&67890", False),
    ],
)
def test_extract_base_id(input_id, output_id, keep_isomer_warts):
    """Test extract_base_id function."""
    check.equal(extract_base_id(input_id, keep_isomer_warts=keep_isomer_warts), output_id)
