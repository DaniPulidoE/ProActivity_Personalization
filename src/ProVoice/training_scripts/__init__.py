"""Study-level training runs: the scripts that produce the thesis numbers.

Three stages, run in order. Each consumes the previous stage's decision file, so
a hyperparameter chosen once cannot silently differ between stages:

  1. ``sweep_population_hparams``  dropout x lr, over 6 fixed validation folds
                                   and 5 seeds -> selected_population.json
                                   (the winning config AND E*, the epoch count)
  2. ``run_lodo_population``       12 leave-one-driver-out population models at
                                   that config, no validation set, no epoch
                                   selection -> 12 checkpoints + a floor table
  3. ``sweep_l2sp_tau``            tau x K over those 12 checkpoints
                                   -> selected_tau.json + the learning curves

Distinct from ``ProVoice.models.*`` (which defines HOW to train) and from
``scripts/`` (one-off tooling): these define the study PROTOCOL — which drivers
are held out where, what is selected on what, and what may be reported.
"""
