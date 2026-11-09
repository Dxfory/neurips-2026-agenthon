"""Track 4 (Explainability) agent, v21.6 (scorer 5.3.0).

Pipeline: load task + dated corpus -> write a safe fallback answer -> plan -> grouped analyst calls
(self-consistency) -> review -> calibrated intervals (interval score) -> verbatim claims -> reasons ->
validated atomic write.
"""

__version__ = "21.14.0"

# v20: the learned local models this image ships, disclosed in the submission's models[] next to the House model
# (track4-analysis-public/docs/ARTIFACT-POLICY.md: "Describe every learned local model with the existing models[]
# entry and access: local"). A plain literal: sub/tools/pack.py reads it without importing the package. The revision
# is t4agent.nowcast.coefficients_digest() (a test keeps them equal); fitting data and cutoff in ARTIFACT_PROVENANCE.md.
LOCAL_MODELS = [
    {"name": "t4agent-cpi-pump-price-nowcast", "version": "21.14.0",
     "revision": "55f9abb9f5c28d9f8a6821860a86b9303fc4ffb95e466293e9d46f96a557f2c7",
     "training_cutoff": "2021-12", "access": "local"},
]
