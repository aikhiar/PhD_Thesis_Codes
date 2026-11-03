# Deployed surrogates of Chapter 6

The study `chapter6` searches on the surrogates deployed in this folder. None is shipped with the
repository: train them with `Frameworks/surrogate_trainer` (`python main.py chapter6`), then copy
everything in `runs/surrogate_chapter6_<key>/surrogate_export/best_models/` (the `.joblib` files and
`model_card.json`) into this folder.

Any regressor trained on the same features (`feature_fn` in `problems/chapter6/model.py`) can be
deployed here in the same way, under the file names of `SURROGATE_FILES` in
`problems/chapter6/parameters.py`. At start-up the study checks the number of features and compares
the library versions recorded in `model_card.json` with those installed.
