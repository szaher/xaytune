from xaytune.data.packing import pack_sequences


class TestPackSequences:
    def test_basic_packing(self):
        sequences = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5]},
            {"input_ids": [6, 7, 8, 9]},
        ]
        packed = pack_sequences(sequences, max_seq_length=6, pad_token_id=0)
        assert len(packed) == 2
        assert len(packed[0]["input_ids"]) == 6
        assert len(packed[1]["input_ids"]) == 6

    def test_sequences_exceeding_max_are_truncated(self):
        sequences = [{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]}]
        packed = pack_sequences(sequences, max_seq_length=4, pad_token_id=0)
        assert len(packed) == 1
        assert len(packed[0]["input_ids"]) == 4

    def test_empty_input(self):
        packed = pack_sequences([], max_seq_length=10, pad_token_id=0)
        assert packed == []

    def test_attention_mask_generated(self):
        sequences = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5]},
        ]
        packed = pack_sequences(sequences, max_seq_length=8, pad_token_id=0)
        assert len(packed) == 1
        mask = packed[0]["attention_mask"]
        assert sum(mask) == 5  # 3 + 2 real tokens
        assert len(mask) == 8

    def test_single_long_sequence(self):
        sequences = [{"input_ids": list(range(100))}]
        packed = pack_sequences(sequences, max_seq_length=50, pad_token_id=0)
        assert len(packed) == 1
        assert len(packed[0]["input_ids"]) == 50

    def test_labels_mask_padding(self):
        sequences = [
            {"input_ids": [1, 2, 3]},
        ]
        packed = pack_sequences(sequences, max_seq_length=6, pad_token_id=0)
        labels = packed[0]["labels"]
        assert labels[:3] == [1, 2, 3]
        assert labels[3:] == [-100, -100, -100]
