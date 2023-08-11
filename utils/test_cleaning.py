import pandas as pd
import pytest

from utils.cleaning import drop_cols_with_quality_threshold, get_snake_case_column_mapping, visualize_missing_values


@pytest.fixture
def sample_data():
    data = {
        'A': [1, 2, None, 4, 5],
        'B': [None, 2, 3, 4, 5],
        'C': [1, None, None, 4, None],
        'D': [1, 2, 3, 4, 5],
        'E': [1, 2, 3, 4, 5]
    }
    return pd.DataFrame(data)


def test_drop_cols_with_quality_threshold(sample_data):
    threshold = 0.3
    result = drop_cols_with_quality_threshold(sample_data, threshold)
    expected_columns = ['A', 'B', 'D', 'E']

    assert list(result.columns) == expected_columns


def test_get_snake_case_column_mapping():
    column_names = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Custom Column Name!',
    ]

    expected_mapping = {
        'Air temperature [K]': 'air_temperature_k',
        'Process temperature [K]': 'process_temperature_k',
        'Rotational speed [rpm]': 'rotational_speed_rpm',
        'Torque [Nm]': 'torque_nm',
        'Custom Column Name!': 'custom_column_name'
    }

    result_mapping = get_snake_case_column_mapping(column_names)

    assert result_mapping == expected_mapping


def test_print_missing_values(sample_data, mocker):
    mocker.patch("matplotlib.pyplot.subplots")
    mocker.patch("seaborn.barplot")
    mocker.patch("matplotlib.pyplot.show")

    result = visualize_missing_values(sample_data)

    expected_columns = ['null_values']
    assert list(result.columns) == expected_columns
