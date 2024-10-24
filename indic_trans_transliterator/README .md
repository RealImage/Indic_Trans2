# Indic-trans: Cross-Language Transliteration Tool

Indic-trans provides a state-of-the-art transliteration module for cross-transliterations (representing multilingual languages in English) among Indian languages, including English and Urdu.

(This deployment is a work in progress and only focuses on text input for the time being)

## Supported Languages

- Hindi
- Bengali
- Gujarati
- Punjabi
- Malayalam
- Kannada
- Tamil
- Telugu
- Oriya
- Marathi
- Assamese
- Konkani
- Bodo
- Nepali
- Urdu
- English

## Key Features

1. **Rule-Based and Machine Learning Models:**
   - Option to choose between rule-based (default) or machine learning-based transliteration.
   
2. **Efficient Lookup:**
   - Build lookup tables to prevent repeated transliteration of the same words, improving performance for large text corpora.
   
3. **K-Best Transliterations:**
   - Generate multiple best transliterations (up to 5) for more accurate results.

## Installation

### Dependencies
- Cython
- SciPy

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/libindic/indic-trans.git
    ```
    or:
    ```bash
    git clone https://github.com/irshadbhat/indic-trans.git
    ```

2. Navigate to the cloned directory:
    ```bash
    cd indic-trans
    ```

3. Install the requirements:
    ```bash
    pip install -r requirements.txt
    pip install .
    ```

## Usage

### 1. From the Console:

   ```bash
   indictrans --s <source_language> --t <target_language> --input <input_file> --output <output_file>
   ```
### Using Python:

```python
from indictrans import Transliterator
trn = Transliterator(source='hin', target='eng', build_lookup=True)
result = trn.transform("input text")
print(result)
```


## Examples

### Hindi to English Example:

**Input (Hindi):**

```plaintext
कांग्रेस पार्टी अध्यक्ष सोनिया गांधी, तमिलनाडु की मुख्यमंत्री जयललिता और रिज़र्व बैंक के गवर्नर रघुराम राजन
```

**Python Script**

```python
from indictrans import Transliterator
trn = Transliterator(source='hin', target='eng', build_lookup=True)
hin_text = "कांग्रेस पार्टी अध्यक्ष सोनिया गांधी, तमिलनाडु की मुख्यमंत्री जयललिता और रिज़र्व बैंक के गवर्नर रघुराम राजन"
result = trn.transform(hin_text)
print(result)
```
**Output**

```plaintext
congress party adhyaksh sonia gandhi, tamilnadu kii mukhyamantri jayalalita aur reserve bank ke governor raghuram rajan
```

### Tamil to English Example:

**Input (Tamil):**

```plaintext
நான் ஒரு மாணவன். இந்தியா ஒரு பெரிய நாடு.
```

**Python Script**

```python
from indictrans import Transliterator
trn = Transliterator(source='tam', target='eng', build_lookup=True)
tam_text = "நான் ஒரு மாணவன். இந்தியா ஒரு பெரிய நாடு."
result = trn.transform(tam_text)
print(result)
```
**Output**

```plaintext
naan oru maanavan. india oru periya naadu.
```

