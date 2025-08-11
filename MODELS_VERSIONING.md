# Model Versioning and Metadata

Purpose: Define a simple, explicit versioning approach and sidecar metadata for all trained models. Applies to both legacy FPL-source models and new FBRef-source models.

## Naming and layout
- Artifacts directory (FBRef): `Data/models_fbref/`
- Recommended structure by target:
  - `Data/models_fbref/<target>/`
  - Examples:
    - `Data/models_fbref/xg/xg_fbref_v1.pkl`
    - `Data/models_fbref/xa/xa_fbref_v1.pkl`
    - `Data/models_fbref/minutes/minutes_fbref_v1.pkl`
- Every model file has a sidecar metadata file named the same + `.meta.json`:
  - `Data/models_fbref/xg/xg_fbref_v1.meta.json`

## Versioning scheme
- Suffix version in filename: `_fbref_v{N}` (start at `v1`, increment on material changes)
- Keep legacy FPL models unchanged in `Data/models/**`
- Do not overwrite prior versions; write new files

## Metadata sidecar (.meta.json)
- One JSON per model artifact
- Must conform to `Data/models_fbref/model_metadata.schema.json`

Required fields:
- model_name: string (e.g., "xg")
- target: enum ["xg","xa","minutes","saves","team_goals_conceded"]
- source: enum ["fbref","fpl"]
- version: string (e.g., "v1")
- schema_version: integer (start with 1)
- created_at: ISO8601 string
- code_commit: string (git SHA)
- model_type: string (e.g., "xgboost", "lightgbm", "sklearn-rf")
- hyperparams: object
- feature_list: string[] (training-time order)
- features_hash: string (sha256 over feature_list + key preprocessing params)
- training_window: { start_season: string, end_season: string, cutoff_date: string }
- data_lineage: {
  db_path: string,
  queries_version: string,          # git SHA of queries.py
  transform_version: string,        # git SHA of transform.py
  gw_mapping_version: string        # hash/sha or date of mapping table
}
- evaluation: {
  overall: { metric: string, value: number }[],
  cohorts: {
    new_to_league?: { metric: string, value: number }[],
    promoted_teams?: { metric: string, value: number }[]
  }
}
- artifacts: { model_path: string }

## Example metadata
```json
{
  "model_name": "xg",
  "target": "xg",
  "source": "fbref",
  "version": "v1",
  "schema_version": 1,
  "created_at": "2025-08-09T12:00:00Z",
  "code_commit": "<git_sha>",
  "model_type": "lightgbm",
  "hyperparams": { "num_leaves": 31, "learning_rate": 0.05 },
  "feature_list": ["shots_per90", "key_passes_per90", "team_attack_strength"],
  "features_hash": "<sha256>",
  "training_window": { "start_season": "2021_22", "end_season": "2024_25", "cutoff_date": "2025-07-31" },
  "data_lineage": {
    "db_path": "/Users/owen/src/Personal/FBRef_DB/master.db",
    "queries_version": "<git_sha>",
    "transform_version": "<git_sha>",
    "gw_mapping_version": "<hash_or_date>"
  },
  "evaluation": {
    "overall": [{ "metric": "RMSE", "value": 0.42 }],
    "cohorts": {
      "new_to_league": [{ "metric": "RMSE", "value": 0.47 }]
    }
  },
  "artifacts": { "model_path": "Data/models_fbref/xg/xg_fbref_v1.pkl" }
}
```
