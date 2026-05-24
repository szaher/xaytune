# Data Pipeline

xaytune's data pipeline handles loading, formatting, tokenizing, packing, and validating training data. The typical flow is:

```
load_dataset → format (automatic) → tokenize_dataset → pack_sequences (optional) → DataLoader(collate_fn)
```

For preference/alignment data, use the preference-specific functions instead:

```
load_preference_dataset → tokenize_preference_dataset → DataLoader(collate_preference)
```

---

## Loading

::: xaytune.data.loader.load_dataset

## Formats

Built-in format functions registered in `format_registry`:

::: xaytune.data.formats.format_alpaca

::: xaytune.data.formats.format_sharegpt

::: xaytune.data.formats.format_chat

::: xaytune.data.formats.format_text

::: xaytune.data.formats.apply_chat_template

## Tokenization

::: xaytune.data.tokenizer.tokenize_dataset

::: xaytune.data.tokenizer.collate_tokenized

## Preference Data

::: xaytune.data.preferences.load_preference_dataset

::: xaytune.data.tokenizer.tokenize_preference_dataset

::: xaytune.data.tokenizer.collate_preference

## Packing

::: xaytune.data.packing.pack_sequences

## Validation

::: xaytune.data.validation.validate_dataset_sample

::: xaytune.data.validation.validate_batch

::: xaytune.data.validation.DataValidationError
