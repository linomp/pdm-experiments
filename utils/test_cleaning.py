import pandas as pd
import pytest

from utils.cleaning import drop_cols_with_quality_threshold, get_snake_case_column_mapping, visualize_missing_values, \
    get_downsampled_df


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


def test_downsampling():
    # Create a sample DataFrame for testing
    data = {
        'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'Failure Type': [0, 0, 0, 1, 1, 2, 2, 2, 2, 3]  # Example classes
    }
    df = pd.DataFrame(data)

    target_name = 'Failure Type'
    df_downsampled = get_downsampled_df(df, target_name)

    # Check if the downsampling resulted in the expected number of samples per class
    unique_classes = df[target_name].unique()
    for failure_type in unique_classes:
        class_count = (df_downsampled[target_name] == failure_type).sum()
        assert class_count == min(df[target_name].value_counts())

    # Check if the resulting DataFrame has the correct number of rows
    expected_rows = len(unique_classes) * min(df[target_name].value_counts())
    assert len(df_downsampled) == expected_rows
