from app.utils import config
from app.utils.config import Config, PathData, load_config, save_config


def test_load_config_valid_file(mocker):
    """Test loading a valid configuration file."""
    mocker.patch(
        'app.utils.config.read_toml',
        return_value={
            'core': {},
            'utils': {
                'log_file_format': '%(asctime)s %(levelname)s %(message)s',
                'log_console_format': '%(message)s %(asctime)s',
                'log_app_format': '%(levelname)s %(message)s %(asctime)s',
                'log_date_format': '%m/%d/%Y %H:%M:%S',
                'text_editor_command': 'vim',
            },
        },
    )
    config_obj = load_config()
    assert isinstance(config_obj, Config)
    assert config_obj.utils.text_editor_command == 'vim'


def test_save_config(mocker, tmp_path):
    """Test saving a configuration file."""
    mocker.patch('app.utils.config.save_toml')
    config_obj = Config(
        core={},
        utils={
            'log_file_format': '%(asctime)s %(levelname)s %(message)s',
            'log_console_format': '%(message)s %(asctime)s',
            'log_app_format': '%(levelname)s %(message)s %(asctime)s',
            'log_date_format': '%m/%d/%Y %H:%M:%S',
            'text_editor_command': '',
        },
    )
    config_file_path = tmp_path / 'config.toml'
    save_config(config_obj, config_file_path)
    config.save_toml(file_path=config_file_path, dict_data=config_obj.model_dump(exclude_defaults=True))

    PathData.config_file_path = config_file_path
    save_config(config_obj)
    config.save_toml(file_path=config_file_path, dict_data=config_obj.model_dump(exclude_defaults=True))


def test_utils_field_validator():
    """Test the field validator for text_editor_command."""
    utils = config.Utils(
        log_file_format='%(asctime)s %(levelname)s %(message)s',
        log_console_format='%(message)s %(asctime)s',
        log_app_format='%(levelname)s %(message)s %(asctime)s',
        log_date_format='%m/%d/%Y %H:%M:%S',
    )
    assert utils.text_editor_command == 'notepad.exe'
