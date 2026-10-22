# Frozen method check for measured differential programs

The [five-person development budget audit](43-differential-search-budget-development.md)
found that a 5,000-program day-1 CV search improved mean day-2 performance
over a random program and the weak fixed differential layout, with large
person-to-person variation. We will run **one frozen rule** on Hyser subjects
06–08 and 17–20. The differential-pair features and search outcomes for
these people have not yet been examined. Their day-2 raw recordings and
monopolar features have been used by earlier experiments, however, so this
is a method-specific check, **not** an untouched participant holdout.

The [frozen runner](../../software/python/scripts/hyser_differential_holdout.py)
will:

1. Verify and cache day-1 digital differences of neighboring channels from
   the publisher-preprocessed forearm array. Generate 5,000 feasible programs
   with seed 42; each uses two adjacent pairs sharing one contact on each of
   four grids, for eight AFE channels and 12 active contacts.
2. Score each program with three-fold, stratified day-1 ridge CV (seed 42,
   alpha 10). Select the maximum CV, with the canonical program order used
   for ties. Write an exclusive selection record with code, source and pair
   map hashes **before creating the day-2 differential feature caches**.
3. Only then extract day-2 differential features and score the selected
   program and the center-fixed differential program using a day-1-trained
   ridge classifier. Score the earlier fixed eight-site monopolar program
   on the same two source sessions as a stronger practical reference.

The primary outcome is the per-person day-2 balanced accuracy difference
between selected and fixed differential programs. We will call this rule a
useful method-level result only if the mean paired gain is at least +0.05,
at least five of seven people improve, no person loses more than 0.05, and
the selected differential cohort mean reaches the fixed monopolar mean.
Report every person even if the rule fails. The outcomes remain forearm
gesture recognition, and digital subtraction does not recreate a physical
front end or establish facial silent-speech behavior.

Because participants and raw day-2 files already exist locally, the
exclusive record and pushed commit prove ordering of this *method's*
selection relative to its differential outcome calculation. They do not
make the evaluation statistically independent of earlier project decisions.
