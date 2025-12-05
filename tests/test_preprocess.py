from mlproject.src.preprocess.offline import OfflinePreprocessor


def test_offline_runs():
    pre = OfflinePreprocessor({
        "preprocessing": {
            "steps": [{'name': 'fill_missing', 'method': 'ffill'}, \
            {'name': 'gen_covariates', 'covariates': {'past':
             ['mobility_inflow'], 'future': ['day_of_week', 'is_holiday', 'influenza_visits'], 'static': ['commune']}}, 
            {'name': 'normalize', 'method': 'zscore', 'columns': None, 'save_artifact': True}]
        }
    })
    df = pre.run()
    assert len(df) > 0

