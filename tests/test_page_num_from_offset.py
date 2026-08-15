from meeplemate.ingest.gamepackage import page_num_from_offset


class TestPageNumFromOffset:
    """Tests for page_num_from_offset."""

    def test_offset_is_page_zero_index(self):
        assert page_num_from_offset(0, 0) == "1"

    def test_pages_after_offset_count_up(self):
        assert [page_num_from_offset(i, 0) for i in range(5)] == [
            "1", "2", "3", "4", "5",
        ]

    def test_pages_before_offset_count_down_through_zero(self):
        assert [page_num_from_offset(i, 2) for i in range(5)] == [
            "-1", "0", "1", "2", "3",
        ]

    def test_offset_at_end_of_range(self):
        assert [page_num_from_offset(i, 4) for i in range(5)] == [
            "-3", "-2", "-1", "0", "1",
        ]
