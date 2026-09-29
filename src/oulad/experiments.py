"""Compare cutoff days and the demographics toggle. Runs are logged, not registered."""
from oulad import registry
from oulad.config import Settings
from oulad.data import TRAIN_TERMS, local_bronze
from oulad.evaluate import classification_metrics, fairness_audit, flatten_fairness
from oulad.features import LABEL, build_features
from oulad.train import DEFAULT_PARAMS, train_model

if __name__ == "__main__":
    s = Settings.from_env()
    registry.setup(s)
    for cutoff in (14, 28, 42):
        df = build_features(local_bronze(s.data_uri), cutoff)
        df = df[df["code_presentation"].isin(TRAIN_TERMS)].reset_index(drop=True)
        for demographics in (False, True):
            trained, va = train_model(df, demographics, s.target_recall)
            m = classification_metrics(va[LABEL], va["proba"], trained.threshold)
            a = fairness_audit(va[LABEL], va["proba"], trained.threshold, va["disability"])
            registry.log_run(
                trained,
                params={**DEFAULT_PARAMS, "cutoff_day": cutoff, "include_demographics": demographics},
                metrics={**{f"val_{k}": v for k, v in m.items()}, **flatten_fairness(a)},
                audit=a,
                tags={"reason": "experiment"},
                model_name=s.model_name,
                register=False,
            )
            print(f"cutoff={cutoff} demographics={demographics} "
                  f"auc={m['roc_auc']:.3f} recall_gap={a['recall_gap']:.3f}")
