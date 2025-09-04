import unittest
from data_prep import clean_text

class TestDataPrep(unittest.TestCase):
    def test_clean_text_basic(self):
        text = "This is SOME fake NEWS!!!"
        cleaned = clean_text(text)
        expected = "fake news"  # stopwords like "this", "is", "some" removed
        self.assertEqual(cleaned, expected)

    def test_clean_text_with_numbers(self):
        text = "Elections 2025 are coming soon!"
        cleaned = clean_text(text)
        expected = "election 2025 come soon"  # lemmatized + cleaned
        self.assertIn("2025", cleaned)        # check number is preserved
        self.assertIn("election", cleaned)

    def test_non_string_input(self):
        self.assertEqual(clean_text(12345), "")

if __name__ == "__main__":
    unittest.main()
