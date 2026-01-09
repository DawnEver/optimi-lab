from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from optimi_lab.utils.file_io import check_path, list_files_in_dir, read_toml, save_toml


@patch('optimi_lab.utils.file_io.Path.open', new_callable=mock_open, read_data='key = "value"')
def test_read_toml_success(mock_file):
    """Test successful TOML file reading."""
    file_path = Path('test_file.toml')
    result = read_toml(file_path)
    assert result == {'key': 'value'}
    mock_file.assert_called_once_with(file_path, encoding='utf-8')


@patch('optimi_lab.utils.file_io.Path.open', new_callable=mock_open)
def test_save_toml_success(mock_file):
    """Test successful TOML file saving."""
    file_path = Path('test_file.toml')
    data = {'key': 'value'}
    save_toml(file_path, data)
    mock_file.assert_called_once_with(file_path, 'w', encoding='utf-8')
    # Ensure write was called with the expected TOML formatted content
    mock_file().write.assert_called_once_with('key = "value"\n')


@patch('optimi_lab.utils.file_io.Path.open', side_effect=OSError)
def test_toml_failure(mock_open):
    """Test failure when saving and reading TOML files."""
    mock_open
    file_path = Path('invalid.toml')
    data = {'key': 'value'}
    with pytest.raises(OSError, match='Failed to save file'):
        save_toml(file_path, data)
    with pytest.raises(OSError, match='Failed to open file'):
        read_toml(file_path)


@patch('optimi_lab.utils.file_io.copyfile')
@patch('optimi_lab.utils.file_io.Path.mkdir')
@patch('optimi_lab.utils.file_io.Path.exists', side_effect=lambda path: path == Path('default_file.txt'))
def test_check_path_create_file(mock_exists, mock_mkdir, mock_copyfile):
    """Test creating file when path does not exist."""
    target_path = Path('test_dir/test_file.txt')
    default_path = Path('default_file.txt')
    # Call check_path function
    check_path(target_path, default_path=default_path, is_dir=False)
    # Verify target path was checked
    mock_exists.assert_any_call(target_path)
    # Verify default path was checked
    mock_exists.assert_any_call(default_path)
    # Verify parent directory of target path was created
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    # Verify copyfile was called to copy default file to target path
    mock_copyfile.assert_called_once_with(default_path, target_path)

    result = check_path(target_path, default_path=None, is_dir=False)
    assert result is False  # Because target_path already exists


@patch('optimi_lab.utils.file_io.Path.mkdir')
@patch('optimi_lab.utils.file_io.Path.exists', return_value=False)
def test_check_path_create_directory(mock_exists, mock_mkdir):
    """Test creating directory when path does not exist."""
    target_path = Path('test_dir')
    result = check_path(target_path, is_dir=True)
    assert result is False
    mock_exists.assert_called_with(target_path)
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@patch('optimi_lab.utils.file_io.os.walk')
@patch('optimi_lab.utils.file_io.check_path', return_value=True)
def test_list_files_in_dir(mock_check_path, mock_walk):
    """Test listing files in directory."""
    dir_path = Path('test_dir')
    mock_walk.return_value = [('test_dir', [], ['file1.txt', 'file2.toml', 'file3.txt'])]
    result = list_files_in_dir(dir_path, file_name_suffix='.txt')
    assert result == ['file1.txt', 'file3.txt']
    mock_check_path.assert_called_once_with(dir_path, is_dir=True)
    mock_walk.assert_called_once_with(dir_path)


@patch('optimi_lab.utils.file_io.check_path', return_value=False)
def test_list_files_in_dir_directory_not_exist(mock_check_path):
    """Test listing files when directory does not exist."""
    dir_path = Path('nonexistent_dir')
    result = list_files_in_dir(dir_path, file_name_suffix='.txt')
    assert result == []  # Returns empty list
    mock_check_path.assert_called_once_with(dir_path, is_dir=True)


if __name__ == '__main__':
    pytest.main([__file__])
