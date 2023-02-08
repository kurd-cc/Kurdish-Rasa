# Kurdish-Rasa
Rasa bot using Kurdish Language Processing Toolkit in messages' processing.

### Installation
1. Create a virtual venv and activate it
2. `pip install -r requirements.txt`
3. Training: `rasa train`
4. Testing: `rasa test -s tests\test_stories.yml`
5. Interactive model training: `rasa interactive`

#### How to use Sorani?
To use Sorani you need to update:
- Update `KLPTTokenizer.py`, change `"Kurmanji", "Latin"` to `"Sorani", "Arabic"`
- Translate `nlu.yml` to Sorani
- For testing purposes, translate `test_stories.yml`