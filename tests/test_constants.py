from GeneticInheritanceGraphLibrary.constants import ValidFlags


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
        i = 0
        for flag in ValidFlags.iedges_combo_standalone_iter():
            i += 1
            assert flag in ValidFlags.IEDGES_COMBO_STANDALONE
        assert i == int.bit_count(ValidFlags.IEDGES_COMBO_STANDALONE)
