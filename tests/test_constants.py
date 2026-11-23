from giglib.constants import ValidFlags


class TestValidFlags:
    def test_valid_flags(self):
        for flag in ValidFlags:
            if flag.name.startswith("IEDGES_"):
                assert flag in ValidFlags.IEDGES_ALL

    def test_gig(self):
        assert ValidFlags.IEDGES_ALL in ValidFlags.GIG
        assert ValidFlags.IEDGES_COMBO_NODE_TABLE in ValidFlags.GIG
        assert ValidFlags.IEDGES_COMBO_STANDALONE in ValidFlags.GIG

    def test_iedges_standalone(self):
        # The standalone requirements are those that don't need the node table
        assert ValidFlags.IEDGES_COMBO_STANDALONE | ValidFlags.IEDGES_COMBO_NODE_TABLE == ValidFlags.IEDGES_ALL
        assert ValidFlags.IEDGES_COMBO_STANDALONE & ValidFlags.IEDGES_COMBO_NODE_TABLE == 0

    def test_iedges_from_tskit(self):
        flags = ValidFlags.IEDGES_FROM_TSKIT
        assert flags in ValidFlags.IEDGES_ALL
        # tskit edges are sorted differently
        assert flags & ValidFlags.IEDGES_SORTED == 0
        assert flags | ValidFlags.IEDGES_SORTED == ValidFlags.IEDGES_ALL
