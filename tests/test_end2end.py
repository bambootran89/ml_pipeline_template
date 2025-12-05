def test_run_pipeline_smoke():
    from mlproject.src.run_pipeline import main

    # run with default config (creates synthetic data) to smoke test
    main(cfg_path=None)
    assert True
