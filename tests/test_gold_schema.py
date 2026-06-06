from dwh.schema.gold_schema import gold_schema


class TestGoldSchema:
    def test_has_expected_fields(self):
        field_names = [f.name for f in gold_schema.fields]
        expected = ["coin_id", "window_start", "window_end", "avg_price",
                     "min_price", "max_price", "avg_volume", "avg_change_pct",
                     "price_volatility", "record_count"]
        for name in expected:
            assert name in field_names

    def test_coin_id_is_not_nullable(self):
        field = gold_schema["coin_id"]
        assert not field.nullable
