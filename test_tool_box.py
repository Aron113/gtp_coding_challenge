import unittest

from tool_box import answer_question


class GhostChainsTests(unittest.TestCase):
    def test_name_question(self) -> None:
        self.assertEqual(answer_question("What is your name?"), "ghost")

    def test_arithmetic_questions(self) -> None:
        self.assertEqual(answer_question("What is 2 + 2?"), 4)
        self.assertEqual(answer_question("What is 9 / 3?"), 3)


if __name__ == "__main__":
    unittest.main()
