"""Проверки сериализации схемы.

Текст схемы попадает в промпт и обязан быть одинаковым во всех прогонах всех
трёх методов. Изменение формата задним числом молча обесценило бы уже
полученные результаты, поэтому формат закреплён строкой в тесте.
"""

from pathlib import Path

import pytest

from kanlora.data.schema import DatabaseSchema, Table, load_schemas, serialize_schema

FIXTURE = Path(__file__).parent / "fixtures" / "tables_mini.json"


@pytest.fixture
def schemas() -> dict[str, DatabaseSchema]:
    return load_schemas(FIXTURE)


def test_loads_every_database(schemas: dict[str, DatabaseSchema]) -> None:
    assert set(schemas) == {"concert_singer", "pets_1"}


def test_columns_are_grouped_by_table(schemas: dict[str, DatabaseSchema]) -> None:
    tables = schemas["concert_singer"].tables
    assert tables == (
        Table("stadium", ("Stadium_ID", "Location", "Name")),
        Table("singer", ("Singer_ID", "Name")),
    )


def test_service_column_is_dropped(schemas: dict[str, DatabaseSchema]) -> None:
    """Столбец * относится ко всей базе, а не к таблице, и в промпт не идёт."""
    for schema in schemas.values():
        for table in schema.tables:
            assert "*" not in table.columns


def test_serialized_format_is_frozen(schemas: dict[str, DatabaseSchema]) -> None:
    assert serialize_schema(schemas["concert_singer"]) == (
        "stadium: Stadium_ID, Location, Name | singer: Singer_ID, Name"
    )


def test_serialization_is_deterministic(schemas: dict[str, DatabaseSchema]) -> None:
    """Порядок берётся из файла и не зависит от порядка обхода словарей."""
    first = serialize_schema(schemas["concert_singer"])
    reloaded = load_schemas(FIXTURE)["concert_singer"]
    assert serialize_schema(reloaded) == first


def test_schema_is_hashable(schemas: dict[str, DatabaseSchema]) -> None:
    """Замороженность — техническая защита от правки схемы в середине прогона."""
    assert len({schemas["pets_1"], schemas["pets_1"]}) == 1
