"""The version scheme.

These assertions are deliberately the same cases as Vesta's
``apps/mobile/lib/router-model/__tests__/version.test.ts``. Two implementations
of one rule only stay in step if they are checked against the same chain.
"""

import unittest

from vesta_router.version import (
    artifact_filename,
    parse_model_version,
    status_implied_by_version,
    status_matches_version,
)


class ParseTests(unittest.TestCase):
    def test_reads_a_release(self):
        v = parse_model_version("1.2.3")
        self.assertIsNotNone(v)
        self.assertEqual((v.major, v.minor, v.patch), (1, 2, 3))
        self.assertIsNone(v.prerelease_tag)
        self.assertEqual(v.raw, "1.2.3")

    def test_reads_both_prerelease_tags(self):
        self.assertEqual(parse_model_version("1.2.0-rc.4").prerelease_tag, "rc")
        self.assertEqual(parse_model_version("1.2.0-rc.4").prerelease_ordinal, 4)
        self.assertEqual(parse_model_version("1.2.0-exp.7").prerelease_tag, "exp")

    def test_refuses_anything_that_is_not_exactly_the_scheme(self):
        # Fail closed. An unparseable version is not "probably newer", and the
        # only alternative to refusing it is ordering it by guesswork.
        for raw in [
            "v1.2.3",
            "1.2",
            "1.2.3.4",
            "1.2.3-beta.1",
            "1.2.3-rc",
            "1.2.3+build5",
            " 1.2.3 ",
            "",
            None,
            3,
        ]:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_model_version(raw))


class OrderingTests(unittest.TestCase):
    ASCENDING = [
        "1.2.0-exp.1",
        "1.2.0-exp.2",
        "1.2.0-rc.1",
        "1.2.0-rc.2",
        "1.2.0",
        "1.2.1",
        "1.3.0",
        "2.0.0",
    ]

    def test_sorts_the_documented_chain(self):
        # As a chain, not as isolated pairs: a comparator can get every adjacent
        # pair right and still not be an order.
        shuffled = [parse_model_version(r) for r in reversed(self.ASCENDING)]
        self.assertEqual([v.raw for v in sorted(shuffled)], self.ASCENDING)

    def test_a_prerelease_sorts_before_its_release(self):
        self.assertLess(parse_model_version("1.2.0-rc.3"), parse_model_version("1.2.0"))
        self.assertGreater(parse_model_version("1.2.0"), parse_model_version("1.2.0-rc.3"))

    def test_equal_versions_are_neither_newer_nor_older(self):
        a, b = parse_model_version("1.2.0"), parse_model_version("1.2.0")
        self.assertFalse(a < b)
        self.assertFalse(a > b)


class StatusTests(unittest.TestCase):
    def test_status_implied_by_shape(self):
        self.assertEqual(status_implied_by_version(parse_model_version("1.0.0")), "stable")
        self.assertEqual(status_implied_by_version(parse_model_version("1.0.0-rc.1")), "candidate")
        self.assertEqual(status_implied_by_version(parse_model_version("1.0.0-exp.1")), "experimental")

    def test_disagreement_is_refused(self):
        self.assertFalse(status_matches_version("stable", parse_model_version("1.0.0-rc.1")))
        self.assertFalse(status_matches_version("candidate", parse_model_version("1.0.0")))
        self.assertFalse(status_matches_version("experimental", parse_model_version("1.0.0-rc.1")))

    def test_terminal_labels_fit_any_shape(self):
        # rejected and superseded are applied to a finished build after the
        # fact, so they say nothing about how it was versioned.
        for raw in ["1.0.0", "1.0.0-rc.1", "1.0.0-exp.1"]:
            for status in ["rejected", "superseded"]:
                self.assertTrue(status_matches_version(status, parse_model_version(raw)))


class FilenameTests(unittest.TestCase):
    def test_derived_from_the_version(self):
        self.assertEqual(artifact_filename("1.2.0"), "vesta-router-1.2.0.cact")
        self.assertEqual(artifact_filename("1.2.0-rc.1"), "vesta-router-1.2.0-rc.1.cact")

    def test_never_the_spikes_ambiguous_name(self):
        self.assertNotEqual(artifact_filename("3.0.0"), "needle3.cact")


if __name__ == "__main__":
    unittest.main()
