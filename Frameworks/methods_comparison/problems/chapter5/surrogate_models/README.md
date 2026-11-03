# Deployed surrogates of Chapter 5

The study `chapter5` searches on the surrogates deployed in this folder. None is shipped with the
repository: train them with `Frameworks/surrogate_trainer` (`python main.py chapter5`), then copy
everything in `runs/surrogate_chapter5_<key>/surrogate_export/best_models/` (the `.joblib` files and
`model_card.json`) into this folder.

Any regressor trained on the same features (`feature_fn` in `problems/chapter5/model.py`) can be
deployed here in the same way, under the file names of `SURROGATE_FILES` in
`problems/chapter5/parameters.py`. At start-up the study checks the number of features and compares
the library versions recorded in `model_card.json` with those installed.
