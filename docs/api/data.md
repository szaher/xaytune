# Data Pipeline

trainlib's data pipeline handles loading, formatting, tokenizing, packing, and validating training data. The typical flow is:

```
load_dataset → format (automatic) → tokenize_dataset → pack_sequences (optional) → DataLoader(collate_fn)
```

For preference/alignment data, use the preference-specific functions instead:

```
load_preference_dataset → tokenize_preference_dataset → DataLoader(collate_preference)
```

---

## Loading

::: trainlib.data.loader.load_dataset

## Formats

Built-in format functions registered in `format_registry`:

::: trainlib.data.formats.format_alpaca

::: trainlib.data.formats.format_sharegpt

::: trainlib.data.formats.format_chat

::: trainlib.data.formats.format_text

::: trainlib.data.formats.apply_chat_template

## Tokenization

::: trainlib.data.tokenizer.tokenize_dataset

::: trainlib.data.tokenizer.collate_tokenized

## Preference Data

::: trainlib.data.preferences.load_preference_dataset

::: trainlib.data.tokenizer.tokenize_preference_dataset

::: trainlib.data.tokenizer.collate_preference

## Packing

::: trainlib.data.packing.pack_sequences

## Validation

::: trainlib.data.validation.validate_dataset_sample

::: trainlib.data.validation.validate_batch

::: trainlib.data.validation.DataValidationError
