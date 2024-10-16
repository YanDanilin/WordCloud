# Preparation

This folder contains python notebooks which show how the data collection was carried out

___./Parsing.ipynb___ - getting alphabet from site with synonyms [synonyms.su](https://synonyms.su/). Parsing procedure takes a lot of time thus the algorithm works for particular letter and requires manual letter changing


___./DatasetForming.ipynb___ - datasets for each letter concatenated here in one dataset with structure:
|   word   | synonyms | synonyms_count |
| :------: | :------: | :------------: |
| $\cdots$ | $\cdots$ |    $\cdots$    |

___./CleaningDataset.ipynb___ - removing dictionary markings, indefinite pronouns; white spaces removed with hyphens (`-`)

|   word   | synonyms | synonyms_count | word_clear | synonyms_clear |
| :------: | :------: | :------------: | :--------: | :------------: |
| $\cdots$ | $\cdots$ |    $\cdots$    |  $\cdots$  |    $\cdots$    |