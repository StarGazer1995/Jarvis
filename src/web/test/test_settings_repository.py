from src.web.settings_repository import UserSettingsRepository


def _payload(suffix: str) -> dict[str, str]:
    return {
        "openai_api_key": f"oa{suffix}",
        "tavily_api_key": f"tv{suffix}",
        "confluence_page_token": f"cf{suffix}",
        "beacon_model_token": f"bc{suffix}",
    }


def test_settings_repository_save_and_load(tmp_path):
    database_url = f"sqlite:///{tmp_path}/settings.db"
    repository = UserSettingsRepository(database_url)
    repository.init_table()

    user_id = "user-1"
    payload = _payload("")

    repository.save_settings(user_id, payload)
    loaded = repository.load_settings(user_id)

    assert loaded == payload


def test_settings_repository_upsert(tmp_path):
    database_url = f"sqlite:///{tmp_path}/settings.db"
    repository = UserSettingsRepository(database_url)
    repository.init_table()

    user_id = "user-2"
    repository.save_settings(user_id, _payload("-1"))
    repository.save_settings(user_id, _payload("-2"))

    loaded = repository.load_settings(user_id)
    assert loaded["openai_api_key"] == "oa-2"
    assert loaded["tavily_api_key"] == "tv-2"
    assert loaded["confluence_page_token"] == "cf-2"
    assert loaded["beacon_model_token"] == "bc-2"
