from src.web.settings_repository import UserSettingsRepository


def test_settings_repository_save_and_load(tmp_path):
    database_url = f"sqlite:///{tmp_path}/settings.db"
    repository = UserSettingsRepository(database_url)
    repository.init_table()

    user_id = "user-1"
    payload = {
        "openai_api_key": "oa",
        "tavily_api_key": "tv",
        "confluence_page_token": "cf",
        "beacon_model_token": "bc",
    }

    repository.save_settings(user_id, payload)
    loaded = repository.load_settings(user_id)

    assert loaded == payload


def test_settings_repository_upsert(tmp_path):
    database_url = f"sqlite:///{tmp_path}/settings.db"
    repository = UserSettingsRepository(database_url)
    repository.init_table()

    user_id = "user-2"
    repository.save_settings(
        user_id,
        {
            "openai_api_key": "oa-1",
            "tavily_api_key": "tv-1",
            "confluence_page_token": "cf-1",
            "beacon_model_token": "bc-1",
        },
    )
    repository.save_settings(
        user_id,
        {
            "openai_api_key": "oa-2",
            "tavily_api_key": "tv-2",
            "confluence_page_token": "cf-2",
            "beacon_model_token": "bc-2",
        },
    )

    loaded = repository.load_settings(user_id)
    assert loaded["openai_api_key"] == "oa-2"
    assert loaded["tavily_api_key"] == "tv-2"
    assert loaded["confluence_page_token"] == "cf-2"
    assert loaded["beacon_model_token"] == "bc-2"
