import pytest

from meeplemate.ingest.ocr import fixup_page_number_sequence


class TestFixupPageNumberSequence:
    """Tests for fixup_page_number_sequence."""

    # --- Edge cases ---

    def test_empty_list(self):
        assert fixup_page_number_sequence([]) == []

    def test_single_element(self):
        assert fixup_page_number_sequence(["5"]) == ["5"]

    def test_single_empty_element(self):
        assert fixup_page_number_sequence([""]) == [""]

    def test_all_empty(self):
        result = fixup_page_number_sequence(["", "", ""])
        assert result == ["1", "2", "3"]

    # --- Arabic numerals: already correct ---

    def test_arabic_already_correct(self):
        assert fixup_page_number_sequence(["1", "2", "3"]) == ["1", "2", "3"]

    def test_arabic_already_correct_starting_higher(self):
        assert fixup_page_number_sequence(["5", "6", "7"]) == ["5", "6", "7"]

    # --- Arabic numerals: fill blanks ---

    def test_arabic_fill_middle_blank(self):
        assert fixup_page_number_sequence(["1", "2", "", "4"]) == ["1", "2", "3", "4"]

    def test_arabic_fill_trailing_blank(self):
        assert fixup_page_number_sequence(["1", "2", "3", ""]) == ["1", "2", "3", "4"]

    def test_arabic_fill_leading_blanks(self):
        result = fixup_page_number_sequence(["", "", "1", "2"])
        assert result == ["-1", "0", "1", "2"]

    def test_arabic_fill_multiple_blanks(self):
        result = fixup_page_number_sequence(["1", "", "", "4"])
        assert result == ["1", "2", "3", "4"]

    def test_arabic_single_anchor_in_middle(self):
        result = fixup_page_number_sequence(["", "", "3", "", ""])
        assert result == ["1", "2", "3", "4", "5"]

    # --- Arabic numerals: fix misreads ---

    def test_arabic_fix_single_misread(self):
        result = fixup_page_number_sequence(["1", "2", "100", "4"])
        assert result == ["1", "2", "3", "4"]

    def test_arabic_fix_misread_at_start(self):
        result = fixup_page_number_sequence(["50", "2", "3", "4"])
        assert result == ["1", "2", "3", "4"]

    def test_arabic_fix_misread_at_end(self):
        result = fixup_page_number_sequence(["1", "2", "3", "99"])
        assert result == ["1", "2", "3", "4"]

    # --- Roman numerals: already correct ---

    def test_roman_already_correct(self):
        assert fixup_page_number_sequence(["i", "ii", "iii"]) == ["i", "ii", "iii"]

    # --- Roman numerals: fill blanks ---

    def test_roman_fill_middle_blank(self):
        result = fixup_page_number_sequence(["i", "ii", "", "iv"])
        assert result == ["i", "ii", "iii", "iv"]

    def test_roman_fill_trailing_blank(self):
        result = fixup_page_number_sequence(["i", "ii", "iii", ""])
        assert result == ["i", "ii", "iii", "iv"]

    # --- Roman numerals: fix misreads ---

    def test_roman_fix_misread(self):
        result = fixup_page_number_sequence(["i", "ii", "x", "iv"])
        assert result == ["i", "ii", "iii", "iv"]

    # --- Leading blanks before roman ---

    def test_leading_blanks_before_roman(self):
        """Negative numbers should be arabic since roman doesn't go negative."""
        result = fixup_page_number_sequence(["", "", "i", "ii"])
        assert result == ["-1", "0", "i", "ii"]

    # --- Duplicate / unfixable sequences ---

    def test_duplicate_page_numbers_preserved(self):
        """Duplicates we can't fix should be left alone; blanks still get filled."""
        result = fixup_page_number_sequence(
            ["", "", "1", "2", "3", "3", "4", "5", "6", ""]
        )
        assert result == ["-1", "0", "1", "2", "3", "3", "4", "5", "6", "7"]

    # --- Mixed scenarios ---

    def test_many_blanks_one_anchor(self):
        result = fixup_page_number_sequence(["", "", "", "4", ""])
        assert result == ["1", "2", "3", "4", "5"]

    def test_longer_sequence_with_misread_and_blanks(self):
        result = fixup_page_number_sequence(["1", "", "3", "100", "5", "", "7"])
        assert result == ["1", "2", "3", "4", "5", "6", "7"]

    def test_all_correct_longer(self):
        seq = [str(i) for i in range(1, 11)]
        assert fixup_page_number_sequence(seq) == seq

    # --- Multi-format sequences ---

    def test_roman_then_arabic(self):
        result = fixup_page_number_sequence(["i", "ii", "iii", "1", "2", "3"])
        assert result == ["i", "ii", "iii", "1", "2", "3"]

    def test_roman_then_arabic_with_blanks(self):
        result = fixup_page_number_sequence(["", "i", "ii", "", "1", "2", "3", ""])
        assert result == ["0", "i", "ii", "iii", "1", "2", "3", "4"]

    def test_roman_then_arabic_fill_gap_between(self):
        """Blanks between format segments are assigned to the preceding segment."""
        result = fixup_page_number_sequence(["i", "ii", "", "", "1", "2"])
        assert result == ["i", "ii", "iii", "iv", "1", "2"]

    def test_roman_then_arabic_with_misreads(self):
        result = fixup_page_number_sequence(["i", "x", "iii", "1", "100", "3"])
        assert result == ["i", "ii", "iii", "1", "2", "3"]

    def test_leading_blanks_roman_then_arabic(self):
        result = fixup_page_number_sequence(["", "", "i", "ii", "1", "2", "3"])
        assert result == ["-1", "0", "i", "ii", "1", "2", "3"]

    def test_arabic_then_roman(self):
        result = fixup_page_number_sequence(["1", "2", "3", "i", "ii", "iii"])
        assert result == ["1", "2", "3", "i", "ii", "iii"]

    # --- Invariants ---

    def test_preserves_list_length(self):
        """The output list must always have the same length as the input."""
        for input_list in [
            ["1", "2", "", "4"],
            ["", "", "1"],
            ["i", "", "iii", "iv", ""],
            ["1", "100", "3"],
            ["i", "ii", "", "1", "2"],
        ]:
            result = fixup_page_number_sequence(input_list)
            assert len(result) == len(input_list), f"Length changed for input {input_list}"
