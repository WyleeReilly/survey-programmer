from survey_programmer.configuration import Configuration


def test_configuration_empty() -> None:
    Configuration.from_context()
